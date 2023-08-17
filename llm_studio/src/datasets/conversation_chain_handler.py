import logging

import numpy as np

from llm_studio.src.datasets.text_utils import get_texts

logger = logging.getLogger(__name__)


class ConversationChainHandler:
    """
    Partitions the dataset into conversation chains.
    The conversation chains consists of a list of conversations, where each conversation round consists of
    a triple of (system, prompt, answer).
    """

    def __init__(self, df, cfg, parser=None):
        self.parser = parser

        if cfg.dataset.parent_id_column != "None":
            assert (
                "id" in df.columns
            ), f"id column required for conversation chaining, DataFrame only has {df.columns}."
            sample_ids = (
                df["id"].astype(df[cfg.dataset.parent_id_column].dtype).tolist()
            )
            parent_ids = df[cfg.dataset.parent_id_column].tolist()

            id2parent_id = {
                id: parent_id
                for id, parent_id in zip(sample_ids, parent_ids)
                if parent_id not in [None, "None"]
                and (
                    not isinstance(parent_id, float)
                    or (not np.isnan(parent_id) and not np.isinf(parent_id))
                )
            }
            if cfg.dataset.limit_chained_samples:
                conversation_start_ids = [
                    idx for idx in sample_ids if idx not in id2parent_id.values()
                ]
            else:
                conversation_start_ids = sample_ids

            conversation_ids_lists = [
                self.get_conversation_ids(id2parent_id, conversation_start_id)
                for conversation_start_id in conversation_start_ids
            ]
            # map from df["id"] to enumeration index
            dataframeid2idx = {id: idx for idx, id in enumerate(sample_ids)}
            self.conversation_ids_lists = [
                [
                    dataframeid2idx[conversation_id]
                    for conversation_id in conversation_ids
                ]
                for conversation_ids in conversation_ids_lists
            ]
        else:
            # no parent id column, so each sample is a conversation chain
            self.conversation_ids_lists = [[idx] for idx in range(len(df))]

        self.prompts = get_texts(df, cfg, separator="")
        if cfg.dataset.answer_column in df.columns:
            self.answers = df[cfg.dataset.answer_column].astype(str).tolist()
        else:
            self.answers = ["" for _ in range(len(self.prompts))]
        self.systems = ["" for _ in range(len(self.prompts))]

        if cfg.dataset.system_column != "None":
            if cfg.dataset.system_column not in df.columns:
                logger.warning(
                    f"System column {cfg.dataset.system_column} not found."
                    f"Disabling functionality."
                )
            else:
                self.systems = df[cfg.dataset.system_column].astype(str).tolist()

    @staticmethod
    def get_conversation_ids(id2parent_id, start_id):
        """
        Gets the conversation chain for a given starting conversation ID.
        Args:
            id2parent_id: A dictionary containing the mapping of IDs to its previous parent ID.
            start_id: The ID of the starting conversation in the chain.
        Returns:
            A list of conversation IDs representing the conversation chain. The chain is ordered from the
            starting conversation to the last conversation in the chain.
        """
        loop_counter = 0  # prevent infinite loops in case of circular parent chains (dataframe issue)

        conversation_chain_ids = [start_id]
        parent_id = start_id
        while parent_id in id2parent_id:
            loop_counter += 1
            # get next parent id
            parent_id = id2parent_id[parent_id]
            conversation_chain_ids.append(parent_id)
            if loop_counter > 1000:
                raise ValueError(
                    f"Parent chain of sample with idx {start_id} exceeds max loop count of 1000. "
                    f"Please ensure that parent chain is not circular."
                )
        return conversation_chain_ids[::-1]

    def __len__(self):
        return len(self.conversation_ids_lists)

    def __getitem__(self, idx):
        """
        Gets a single conversation chain.
        The conversation may be:
        - a single (system, prompt, answer) round,
          if cfg.dataset.parent_id_column == "None" or there is no parent_id for the conversation
        - a conversation potentially starting somewhere in the middle of the conversation,
          if the conversation is chained and limit_chained_samples is set to False
        - always a complete conversation, if the conversation is chained
          and limit_chained_samples is True

        """
        prompts = [self.prompts[i] for i in self.conversation_ids_lists[idx]]
        answers = [self.answers[i] for i in self.conversation_ids_lists[idx]]
        systems = [self.systems[i] for i in self.conversation_ids_lists[idx]]
        text_dict = {
            "prompts": prompts,
            "answers": answers,
            "systems": systems,
        }
        if self.parser:
            text_dict = self.parser(text_dict)
        return text_dict
