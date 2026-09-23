//! The module of the arena collection.
//!
//! See [`Arenas`] for information.

mod bitmap;

use core::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::{self, ManuallyDrop, MaybeUninit},
    num::NonZeroUsize,
    ops::Range,
    panic,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering::*},
};

use self::bitmap::Bitmap;
use crate::{
    base::{BaseAlloc, Chunk},
    config::{DIRTY_SLAB_CAP, SLAB_SHIFT, SLAB_SIZE},
    heap::ObjSizeType,
    slab::{Slab, SlabRef, SlabSource},
};

const BYTE_WIDTH: usize = u8::BITS as usize;

pub(crate) const fn slab_layout(n: usize) -> Layout {
    match Layout::from_size_align(n << SLAB_SHIFT, SLAB_SIZE) {
        Ok(layout) => layout,
        Err(_) => panic!("invalid slab layout"),
    }
}

struct Arena<B: BaseAlloc> {
    arena_id: NonZeroUsize,
    chunk: Chunk<B>,
    header: Chunk<B>,
    is_exclusive: bool,
    slab_count: usize,
    search_index: AtomicUsize,
    dirty_count: AtomicUsize,
}

impl<B: BaseAlloc> Arena<B> {
    const LAYOUT: Layout = Layout::new::<Self>();

    fn header_layout(slab_count: usize, is_exclusive: bool) -> Layout {
        if !is_exclusive {
            let bitmap_size = slab_count
                .div_ceil(BYTE_WIDTH)
                .next_multiple_of(mem::size_of::<AtomicUsize>());
            // Two parallel bitmaps: allocation state, then dirty state.
            let n = 2 * (bitmap_size / mem::size_of::<AtomicUsize>());

            let bitmap_layout = Layout::array::<AtomicUsize>(n).unwrap();
            let (header_layout, offset) = Self::LAYOUT.extend(bitmap_layout).unwrap();
            assert_eq!(offset, Self::LAYOUT.size());
            header_layout
        } else {
            Self::LAYOUT
        }
    }

    /// # Safety
    ///
    /// `header` must contains a valid bitmap covering exact all memory blocks
    /// in `chunk` (i.e. `slab_count`).
    unsafe fn from_dynamic<'a>(
        header: Chunk<B>,
        chunk: Chunk<B>,
        is_exclusive: bool,
        slab_count: usize,
    ) -> &'a mut Arena<B> {
        // SAFETY: The pointer is properly aligned.
        let arena = unsafe {
            let pointer = header.pointer().cast::<Arena<B>>();
            pointer.as_uninit_mut().write(Arena {
                arena_id: NonZeroUsize::MAX,
                chunk,
                header,
                is_exclusive,
                slab_count,
                search_index: Default::default(),
                dirty_count: Default::default(),
            })
        };

        if !is_exclusive {
            // SAFETY: the bitmap pointers point to valid & uninit memory blocks.
            unsafe {
                let maybe = arena.bitmap_ptr().as_uninit_slice_mut();
                maybe.fill(MaybeUninit::new(0));
                let maybe = arena.dirty_bitmap_ptr().as_uninit_slice_mut();
                maybe.fill(MaybeUninit::new(0));
            }
            let bitmap = arena.bitmap();
            bitmap.set::<true>(slab_count.try_into().unwrap()..bitmap.len());
        }

        arena
    }

    fn new<'a>(
        base: &B,
        slab_count: usize,
        align: Option<usize>,
        is_exclusive: bool,
    ) -> Result<&'a mut Self, Error<B>> {
        let layout = match align {
            Some(align) => slab_layout(slab_count)
                .align_to(align)
                .expect("invalid align"),
            None => slab_layout(slab_count),
        };

        let header_layout = Self::header_layout(slab_count, is_exclusive);

        let chunk = base.allocate(layout, is_exclusive).map_err(Error::Alloc)?;
        let header = base.allocate(header_layout, true).map_err(Error::Alloc)?;

        // SAFETY: `header` is valid.
        Ok(unsafe { Self::from_dynamic(header, chunk, is_exclusive, slab_count) })
    }

    fn new_chunk<'a>(base: &B, chunk: Chunk<B>) -> Result<&'a mut Self, Error<B>> {
        let size = chunk.pointer().len();
        let slab_count = size / SLAB_SIZE;
        debug_assert!(chunk.pointer().is_aligned_to(SLAB_SIZE));

        let header_layout = Self::header_layout(slab_count, false);
        let header = base.allocate(header_layout, true).map_err(Error::Alloc)?;

        Ok(unsafe { Self::from_dynamic(header, chunk, false, slab_count) })
    }

    /// # Safety
    ///
    /// `arena` must have no other references alive.
    unsafe fn drop(arena: NonNull<Self>) {
        // SAFETY: We read the data first so as to avoid dropping the dropped data.
        drop(unsafe { arena.read() });
    }

    fn bitmap_size(&self) -> usize {
        self.slab_count
            .div_ceil(BYTE_WIDTH)
            .next_multiple_of(mem::size_of::<AtomicUsize>())
    }

    fn bitmap_ptr(&self) -> NonNull<[u8]> {
        let (ptr, _) = self.header.pointer().to_raw_parts();
        NonNull::from_raw_parts(
            ptr.map_addr(|addr| addr.checked_add(Self::LAYOUT.size()).unwrap()),
            self.bitmap_size(),
        )
    }

    fn dirty_bitmap_ptr(&self) -> NonNull<[u8]> {
        let size = self.bitmap_size();
        let (ptr, _) = self.header.pointer().to_raw_parts();
        NonNull::from_raw_parts(
            ptr.map_addr(|addr| addr.checked_add(Self::LAYOUT.size() + size).unwrap()),
            size,
        )
    }

    fn bitmap_from(&self, ptr: NonNull<[u8]>) -> &Bitmap {
        let (ptr, len) = ptr.to_raw_parts();
        let slice = NonNull::from_raw_parts(ptr, len / mem::size_of::<AtomicUsize>());
        // SAFETY: The bitmap pointer points to a valid `[AtomicUsize]`.
        Bitmap::new(unsafe { slice.as_ref() })
    }

    fn bitmap(&self) -> &Bitmap {
        self.bitmap_from(self.bitmap_ptr())
    }

    /// Tracks freed slabs whose decommit was skipped; their memory is stale.
    fn dirty_bitmap(&self) -> &Bitmap {
        self.bitmap_from(self.dirty_bitmap_ptr())
    }

    /// # Safety
    ///
    /// `raw` must be within this arena's chunk.
    unsafe fn slab_range(&self, raw: NonNull<[u8]>) -> Range<u32> {
        let (ptr, len) = raw.to_raw_parts();
        let offset = unsafe {
            ptr.cast::<u8>()
                .offset_from_unsigned(self.chunk.pointer().cast())
        };
        (offset / SLAB_SIZE) as u32..((offset + len) / SLAB_SIZE) as u32
    }

    fn allocate_slices(&self, count: usize) -> Option<NonNull<[u8]>> {
        let start = self.search_index.load(Relaxed);
        let (idx, bit) = self.bitmap().allocate(start, count.try_into().ok()?)?;
        self.search_index.store(idx, Relaxed);

        // `idx` indexes usize words; `BYTE_WIDTH` sizes the bitmap in bytes. Mixing them placed
        // every slab from the second word 56 slabs too low, on top of live ones.
        let offset = (idx * usize::BITS as usize + (bit as usize)) * SLAB_SIZE;
        // SAFETY: `idx` and `bit` is valid, and thus `offset` is within the chunk
        // memory range.
        let data = unsafe { self.chunk.pointer().byte_add(offset) };
        Some(NonNull::from_raw_parts(
            data.cast::<()>(),
            SLAB_SIZE * count,
        ))
    }

    #[cfg(debug_assertions)]
    fn check_ptr(&self, ptr: NonNull<u8>) -> bool {
        let addr = ptr.as_ptr().addr();
        let (origin, size) = self.chunk.pointer().to_raw_parts();
        let origin = origin.as_ptr().addr();
        (origin..origin + size).contains(&addr)
    }

    fn allocate(
        &self,
        thread_id: usize,
        count: usize,
        base: &B,
    ) -> Option<Result<SlabRef<'_, B>, Error<B>>> {
        let ptr = self.allocate_slices(count)?;

        let (addr, _) = ptr.to_raw_parts();
        if let Err(err) =
            unsafe { base.commit(NonNull::from_raw_parts(addr, mem::size_of::<Slab<'_, B>>())) }
        {
            return Some(Err(Error::Commit(err)));
        }
        // A dirty slab skipped its decommit in `deallocate`, so its free
        // memory is stale and must not claim the base's zero guarantee.
        // SAFETY: `ptr` was just allocated from this arena's chunk.
        let dirty = self
            .dirty_bitmap()
            .clear_count(unsafe { self.slab_range(ptr) });
        if dirty > 0 {
            self.dirty_count.fetch_sub(dirty as usize, Relaxed);
        }
        // SAFETY: The fresh allocation is aligned to `SLAB_SIZE`.
        Some(Ok(unsafe {
            Slab::init(
                ptr,
                thread_id,
                SlabSource::Arena(self.arena_id),
                B::IS_ZEROED && dirty == 0,
            )
        }))
    }

    /// # Safety
    ///
    /// - `slab` must be previously allocated from this arena;
    /// - No more references to the `slab` or its shards exist after calling
    ///   this function.
    unsafe fn deallocate(&self, slab: SlabRef<'_, B>, base: &B) -> usize {
        let raw = slab.into_raw();
        // SAFETY: `raw` was allocated from this arena's chunk.
        let range = unsafe { self.slab_range(raw) };
        let count = (range.end - range.start) as usize;
        // Keep up to `DIRTY_SLAB_CAP` freed slabs committed per arena, marked
        // dirty instead of decommitted: a promptly reused slab would only be
        // re-faulted and re-zeroed. Dirty bits are set before the allocation
        // bits are cleared, since clearing those publishes the slabs to
        // `allocate_slices`.
        let cache = self
            .dirty_count
            .fetch_update(Relaxed, Relaxed, |n| {
                (n + count <= DIRTY_SLAB_CAP).then_some(n + count)
            })
            .is_ok();
        if cache {
            self.dirty_bitmap().set::<true>(range.clone());
        } else {
            unsafe { base.decommit(raw) };
        }
        self.bitmap().set::<false>(range);
        count
    }
}

const MAX_ARENAS: usize = 256;
/// A collection of arenas.
///
/// This structure manages all the memory within its lifetime. Multiple
/// [`Context`](crate::heap::Context)s and [`Heap`](crate::heap::Heap)s can have
/// reference to one instance of this type.
///
/// See [the crate-level documentation](crate) for its usage.
pub struct Arenas<B: BaseAlloc> {
    base: B,
    arenas: [AtomicPtr<Arena<B>>; MAX_ARENAS],
    arena_count: AtomicUsize,
    abandoned: AtomicPtr<()>,
    abandoned_visited: AtomicPtr<()>,
}

impl<B: BaseAlloc> Arenas<B> {
    // We're using this constant to initialize the array, so no real manipulation on
    // this constant is performed.
    #[allow(clippy::declare_interior_mutable_const)]
    const ARENA_INIT: AtomicPtr<Arena<B>> = AtomicPtr::new(ptr::null_mut());

    /// Creates a new collection of arenas.
    pub const fn new(base: B) -> Self {
        Arenas {
            base,
            arenas: [Self::ARENA_INIT; MAX_ARENAS],
            arena_count: AtomicUsize::new(0),
            // slab_count: AtomicUsize::new(0),
            abandoned: AtomicPtr::new(ptr::null_mut()),
            abandoned_visited: AtomicPtr::new(ptr::null_mut()),
        }
    }

    fn push_arena<'a>(&'a self, arena: &'a mut Arena<B>) -> Result<&'a Arena<B>, Error<B>> {
        let mut index = self.arena_count.load(Relaxed);
        loop {
            break if index < MAX_ARENAS {
                if let Err(i) = self
                    .arena_count
                    .compare_exchange(index, index + 1, AcqRel, Acquire)
                {
                    index = i;
                    continue;
                }
                // SAFETY: 1 <= index + 1 <= MAX_ARENAS
                arena.arena_id = unsafe { NonZeroUsize::new_unchecked(index + 1) };
                if self.arenas[index]
                    .compare_exchange(ptr::null_mut(), arena, AcqRel, Acquire)
                    .is_err()
                {
                    index = self.arena_count.load(Relaxed);
                    continue;
                }
                Ok(arena)
            } else if self.arenas.iter().enumerate().any(|(index, slot)| {
                // SAFETY: 1 <= index + 1 <= MAX_ARENAS
                arena.arena_id = unsafe { NonZeroUsize::new_unchecked(index + 1) };
                slot.compare_exchange(ptr::null_mut(), arena, AcqRel, Acquire)
                    .is_ok()
            }) {
                Ok(arena)
            } else {
                // SAFETY: The arena is freshly allocated.
                unsafe { Arena::drop(arena.into()) };
                Err(Error::ArenaExhausted)
            };
        }
    }

    fn push_abandoned(&self, slab: SlabRef<'_, B>) {
        debug_assert!(slab.is_abandoned());
        let mut next = self.abandoned.load(Relaxed);
        loop {
            slab.abandoned_next.store(next, Relaxed);
            match self.abandoned.compare_exchange_weak(
                next,
                slab.as_ptr().as_ptr(),
                AcqRel,
                Acquire,
            ) {
                Ok(_) => break,
                Err(e) => next = e,
            }
        }
    }

    fn push_abandoned_visited(&self, slab: SlabRef<'_, B>) {
        let mut next = self.abandoned_visited.load(Relaxed);
        loop {
            slab.abandoned_next.store(next, Relaxed);
            match self.abandoned_visited.compare_exchange_weak(
                next,
                slab.as_ptr().as_ptr(),
                AcqRel,
                Acquire,
            ) {
                Ok(_) => break,
                Err(e) => next = e,
            }
        }
    }

    fn reappend_abandoned_visited(&self) -> bool {
        if self.abandoned_visited.load(Relaxed).is_null() {
            return false;
        }
        let first = self.abandoned_visited.swap(ptr::null_mut(), AcqRel);
        if first.is_null() {
            return false;
        };

        if self.abandoned.load(Relaxed).is_null()
            && self
                .abandoned
                .compare_exchange(ptr::null_mut(), first, AcqRel, Acquire)
                .is_ok()
        {
            return true;
        }

        let mut last = first;
        let last = loop {
            let next = unsafe { (*last.cast::<Slab<'_, B>>()).abandoned_next.load(Relaxed) };
            last = match next.is_null() {
                true => break last,
                false => next,
            };
        };

        let mut next = self.abandoned.load(Relaxed);
        loop {
            unsafe {
                (*last.cast::<Slab<'_, B>>())
                    .abandoned_next
                    .store(next, Relaxed)
            };
            match self
                .abandoned
                .compare_exchange_weak(next, first, AcqRel, Acquire)
            {
                Ok(_) => break,
                Err(e) => next = e,
            }
        }

        true
    }

    fn pop_abandoned(&self) -> Option<SlabRef<'_, B>> {
        let mut next = self.abandoned.load(Relaxed);
        if next.is_null() && !self.reappend_abandoned_visited() {
            return None;
        }
        next = self.abandoned.load(Relaxed);
        let ret = loop {
            let ptr = NonNull::new(next)?;
            let new_next = unsafe { (*next.cast::<Slab<'_, B>>()).abandoned_next.load(Relaxed) };
            match self
                .abandoned
                .compare_exchange_weak(next, new_next, AcqRel, Acquire)
            {
                Ok(_) => break unsafe { SlabRef::from_ptr(ptr) },
                Err(e) => next = e,
            }
        };
        ret.abandoned_next.store(ptr::null_mut(), Release);
        Some(ret)
    }

    pub(crate) fn reclaim_all(&self) {
        while let Some(slab) = self.pop_abandoned() {
            if slab.collect_abandoned() {
                unsafe { self.deallocate(slab) };
            } else {
                self.push_abandoned_visited(slab);
            }
        }
    }

    fn try_reclaim(&self, thread_id: usize, count: usize, align: usize) -> Option<SlabRef<'_, B>> {
        const MAX_TRIAL: usize = 8;
        let mut trial = MAX_TRIAL;
        while trial > 0
            && let Some(slab) = self.pop_abandoned()
        {
            if slab.collect_abandoned() {
                if slab.size >= count * SLAB_SIZE && slab.as_ptr().is_aligned_to(align) {
                    let slab_source = match &slab.source {
                        &SlabSource::Arena(aid) => SlabSource::Arena(aid),
                        SlabSource::Base { chunk } => SlabSource::Base {
                            chunk: UnsafeCell::new(unsafe { ptr::read(chunk.get()) }),
                        },
                    };
                    let raw = slab.into_raw();
                    let slab = unsafe { Slab::init(raw, thread_id, slab_source, false) };
                    return Some(slab);
                } else {
                    unsafe { self.deallocate(slab) };
                }
            } else {
                self.push_abandoned_visited(slab);
            }
            trial -= 1;
        }
        None
    }

    fn all_arenas(&self) -> impl Iterator<Item = (usize, &Arena<B>)> {
        let iter = self.arenas[..self.arena_count.load(Acquire)].iter();
        // SAFETY: We check the nullity of the pointers.
        iter.enumerate()
            .filter_map(|(index, arena)| Some((index, unsafe { arena.load(Acquire).as_ref() }?)))
    }

    fn arenas(&self, is_exclusive: bool) -> impl Iterator<Item = (usize, &Arena<B>)> {
        self.all_arenas()
            .filter(move |(_, arena)| arena.is_exclusive == is_exclusive)
    }

    /// Retrieves the base allocator of this arena collection.
    pub const fn base(&self) -> &B {
        &self.base
    }

    /// Manages another chunk previously allocated by an instance of its base
    /// allocator.
    ///
    /// This function creates a new arena from the chunk and push it to the
    /// collection for further allocation, extending the heap's overall
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the header allocation has failed, or the collection
    /// is full of arenas.
    pub fn manage(&self, chunk: Chunk<B>) -> Result<(), Error<B>> {
        let arena = Arena::new_chunk(&self.base, chunk)?;
        self.push_arena(arena)?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    pub(crate) fn check_ptr(&self, ptr: NonNull<u8>) -> bool {
        self.all_arenas().any(|(_, arena)| arena.check_ptr(ptr))
    }

    pub(crate) fn allocate(
        &self,
        thread_id: usize,
        count: NonZeroUsize,
        align: usize,
        direct: bool,
    ) -> Result<SlabRef<'_, B>, Error<B>> {
        let count = count.get();

        if let Some(reclaimed) = self.try_reclaim(thread_id, count, align) {
            return Ok(reclaimed);
        }

        if align <= ObjSizeType::LARGE_MAX {
            if let Some(slab) = self
                .arenas(false)
                .find_map(|(_, arena)| arena.allocate(thread_id, count, &self.base))
            {
                return slab;
            }
            const MIN_RESERVE_COUNT: usize = 32;

            let reserve_count = count.max(MIN_RESERVE_COUNT);
            let arena = Arena::new(&self.base, reserve_count, None, false)?;
            return match self.push_arena(arena) {
                Ok(arena) => arena.allocate(thread_id, count, &self.base).unwrap(),
                _ if direct => {
                    let res = self.base().allocate(slab_layout(reserve_count), true);
                    let chunk = res.map_err(Error::Alloc)?;
                    let ptr = chunk.pointer();
                    let source = SlabSource::Base {
                        chunk: UnsafeCell::new(ManuallyDrop::new(chunk)),
                    };
                    let slab = unsafe { Slab::init(ptr, thread_id, source, B::IS_ZEROED) };
                    Ok(slab)
                }
                Err(err) => Err(err),
            };
        }

        let layout = slab_layout(count).align_to(align).unwrap();
        Err(Error::Unsupported(layout))
    }

    /// # Safety
    ///
    /// `slab` must be previously allocated from this structure;
    pub(crate) unsafe fn deallocate(&self, slab: SlabRef<'_, B>) {
        if !slab.is_abandoned() {
            match &slab.source {
                &SlabSource::Arena(id) => {
                    let arena = self.arenas[id.get() - 1].load(Acquire);
                    debug_assert!(!arena.is_null());
                    // SAFETY: `arena` is obtained from the unique `arena_id`, and the arena won't
                    // be dropped as long as any allocation from it is alive.
                    let _slab_count = unsafe { (*arena).deallocate(slab, &self.base) };
                }
                SlabSource::Base { chunk } => {
                    let chunk = unsafe { ptr::read(chunk.get()) };
                    #[warn(clippy::forget_non_drop)]
                    mem::forget(slab);
                    drop(ManuallyDrop::into_inner(chunk));
                }
            }
        } else {
            self.push_abandoned(slab)
        }
    }
}

impl<B: BaseAlloc> Drop for Arenas<B> {
    fn drop(&mut self) {
        let iter = self.arenas[..*self.arena_count.get_mut()].iter_mut();
        iter.filter_map(|arena| NonNull::new(mem::replace(arena.get_mut(), ptr::null_mut())))
            // SAFETY: All the arenas are unreferenced due to the lifetime model.
            .for_each(|arena| unsafe { Arena::drop(arena) })
    }
}

/// The errors of all the functions of this crate.
#[derive(Debug)]
pub enum Error<B: BaseAlloc> {
    /// The base error returned when an allocation failed.
    Alloc(B::Error),
    /// The base error returned when an commission failed.
    Commit(B::Error),
    /// The arena collection is full of arenas.
    ArenaExhausted,
    /// Allocations for this layout is not yet supported.
    Unsupported(Layout),
}

#[cfg(feature = "base-mmap")]
const _: () = assert!(!mem::needs_drop::<Error<crate::base::Mmap>>());

impl<B: BaseAlloc> core::fmt::Display for Error<B>
where
    B::Error: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Alloc(err) => write!(f, "base allocation failed: {err}"),
            Error::Commit(err) => write!(f, "base allocator failed for committing memory: {err}"),
            Error::ArenaExhausted => write!(f, "the arena collection is full of arenas"),
            Error::Unsupported(layout) => write!(f, "unsupported layout: {layout:?}"),
        }
    }
}
