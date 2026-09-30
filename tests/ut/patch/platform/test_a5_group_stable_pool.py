# SPDX-License-Identifier: Apache-2.0
"""A5 packed-page allocation must preserve reusable prefix blocks."""

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _GroupStableBlockPool


def _pool(num_gpu_blocks: int) -> _GroupStableBlockPool:
    return _GroupStableBlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=128,
    )


def test_unowned_pages_are_used_before_recycling_group_pages():
    pool = _pool(4)
    blocks = pool.get_new_blocks_for_group(2, group_id=0)
    pool.free_blocks_for_group(blocks, group_id=0)

    fresh = pool.get_new_blocks_for_group(1, group_id=0)

    assert fresh[0].block_id == 3


def test_touched_page_rejoins_at_tail_without_stale_queue_entry():
    pool = _pool(3)
    oldest, newer = pool.get_new_blocks_for_group(2, group_id=0)
    pool.free_blocks_for_group([oldest, newer], group_id=0)

    pool.touch([oldest])
    pool.free_blocks_for_group([oldest], group_id=0)

    recycled = pool.get_new_blocks_for_group(1, group_id=0)

    assert recycled[0] is newer
