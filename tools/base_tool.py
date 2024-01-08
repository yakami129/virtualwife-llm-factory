from abc import ABC, abstractmethod
from typing import Optional


class BaseTool(ABC):

    @abstractmethod
    def run(self, args: Optional[dict], **kwargs):
        pass
