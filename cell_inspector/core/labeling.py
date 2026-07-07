"""标注历史与撤销。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .metadata import LABEL_BLOOD, LABEL_FETAL, LABEL_SKIP


@dataclass
class LabelAction:
    cell_id: str
    prev_label: int
    new_label: int
    idx: int


@dataclass
class LabelHistory:
    """内存中的标注撤销栈与最近样本索引。"""

    undo_stack: list[LabelAction] = field(default_factory=list)
    recent_fetal: list[int] = field(default_factory=list)
    recent_blood: list[int] = field(default_factory=list)
    max_recent: int = 10

    def record(
        self,
        cell_id: str,
        prev_label: int,
        new_label: int,
        idx: int,
    ) -> None:
        self.undo_stack.append(
            LabelAction(cell_id=cell_id, prev_label=prev_label, new_label=new_label, idx=idx)
        )
        if new_label == LABEL_FETAL:
            self._push_recent(self.recent_fetal, idx)
        elif new_label == LABEL_BLOOD:
            self._push_recent(self.recent_blood, idx)

    def _push_recent(self, lst: list[int], idx: int) -> None:
        if idx in lst:
            lst.remove(idx)
        lst.insert(0, idx)
        del lst[self.max_recent :]

    def pop_undo(self) -> Optional[LabelAction]:
        return self.undo_stack.pop() if self.undo_stack else None

    def clear(self) -> None:
        self.undo_stack.clear()
        self.recent_fetal.clear()
        self.recent_blood.clear()
