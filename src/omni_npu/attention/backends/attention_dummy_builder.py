# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod, ABCMeta


class DummyAttentionMetadataBuilder(metaclass=ABCMeta):
    """
    Attention metadata builder interface for building dummy metadata.
    """

    @abstractmethod
    def build_dummy(self, *args, **kwargs):
        pass

    @abstractmethod
    def mark_static_for_attn_metadata(self, *args, **kwargs):
        pass
