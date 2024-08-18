import logging
from typing import Dict, List

import numpy as np

from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.utils import PatchedAttribute

logger = logging.getLogger(__name__)


class ConversationChainHandler:
    """
    This class partitions the dataset into chains of conversations.
    Each chain is comprised of a list of conversation rounds.
    Each round within a conversation is represented as a triplet:
     (system, prompt, answer).

    The resulting structure of the chains is conditional on
    the DataFrame's structure and configuration:

    - Without a 'parent_id' in the DataFrame, each conversation chain is a single round.
     So, for every `i`-th row in the DataFrame, 0 <= `i` < len(df),
     the chain would look like: [(system_i, prompt_i, answer_i)]

    - With a 'parent_id' in the DataFrame and
      if `cfg.dataset.limit_chained_samples` is set to False,
      each chain encapsulates all preceding conversations
      for every `i`-th row in the DataFrame,
      0 <= `i` < len(df).
      The resultant chain would take shape:
          [(system_start_conversation_i,
            prompt_start_conversation_i,
            answer_start_conversation_i),
           ...,
           (system_i, prompt_i, answer_i)]

    - With a 'parent_id' in the DataFrame and
      if `cfg.dataset.limit_chained_samples` is set to True,
      each conversation chain incorporates only full conversations.
      The chain hence condenses into:
          [(system_start_conversation_i,
            prompt_start_conversation_i,
            answer_start_conversation_i),
           ...,
          (system_end_conversation_i,
           prompt_end_conversation_i,
           answer_end_conversation_i)]
      where `i` represents complete conversations only.
    """

    def __init__(
        self,
        df,
        cfg,
    ):
        # Do not set self.cfg = cfg, as ConversationChainHandler
        # will be used with PatchedAttribute context manager.
        self.conversation_chain_ids = self.get_conversation_chain_ids(cfg, df)
        self.prompts = get_texts(df, cfg)
        self.answers = self.get_answers(df, cfg)
        self.systems = self.get_systems(cfg, df)

    def get_conversation_chain_ids(self, cfg, df):
        """
        Gets the conversation chain IDs for the given DataFrame.
        E.g. if conversation_chain_ids = [[13, 44, 8], ...],
        then the first conversation chain consists of
        [df.iloc[13], df.iloc[44], df.iloc[8]]
        with
            - df.iloc[13] denotes the first round of the conversation
            - df.iloc[44] denotes the second round of the conversation
            - df.iloc[8] denotes the end of the conversation
        if limit_chained_samples is True, df.iloc[13] will have no parent_id,
        i.e. it is the start of the conversation.
        """
        if (
            cfg.dataset.parent_id_column in ["None", None]
            # Handle case where train Dataframe has conversation chains,
            # but val Dataframe does not
            or cfg.dataset.parent_id_column not in df.columns
        ):
            # no parent id column, so each triplet (system_i, prompt_i, answer_i)
            # is a conversation chain
            return [[idx] for idx in range(len(df))]

        assert "id" in df.columns, (
            f"id column required for conversation chaining, "
            f"DataFrame only has {df.columns}."
        )
        # sample and parent ids can have any dtype, such as str, int, float, etc.
        # id column can be int, while parent_id column can be float
        # (as some values are NaN) so we cast id to the same dtype
        sample_ids = df["id"].astype(df[cfg.dataset.parent_id_column].dtype).tolist()
        parent_ids = df[cfg.dataset.parent_id_column].tolist()
        # Some datasets may include parent ids that are not in the dataset.
        sample_ids_set = set(sample_ids)
        parent_ids = [idx if idx in sample_ids_set else "None" for idx in parent_ids]

        id2parent_id = {
            idx: parent_id
            for idx, parent_id in zip(sample_ids, parent_ids)
            if parent_id not in [None, "None"]
            and (
                not isinstance(parent_id, float)
                or (not np.isnan(parent_id) and not np.isinf(parent_id))
            )
        }
        if cfg.dataset.limit_chained_samples:
            # end id == id is not a parent id of another conversation id
            valid_parent_ids = set(id2parent_id.values())
            conversation_end_ids = [
                idx for idx in sample_ids if idx not in valid_parent_ids
            ]
        else:
            conversation_end_ids = sample_ids
        conversation_chain_ids = [
            self.get_conversation_ids(id2parent_id, conversation_end_id)
            for conversation_end_id in conversation_end_ids
        ]
        # map from df["id"] to enumeration index
        dataframeid2idx = {id: idx for idx, id in enumerate(sample_ids)}
        conversation_chain_ids = [
            [dataframeid2idx[conversation_id] for conversation_id in conversation_ids]
            for conversation_ids in conversation_chain_ids
        ]
        return conversation_chain_ids

    def get_answers(self, df, cfg):
        answer_column = cfg.dataset.answer_column
        if isinstance(answer_column, (list, tuple)):
            answers = []
            for col in answer_column:
                if col in df.columns:
                    answers.append(df[col].astype(str).tolist())
                else:
                    answers.append(["" for _ in range(len(self.prompts))])
            answers = [",".join(ans) for ans in zip(*answers)]
        else:
            if answer_column in df.columns:
                answers = df[answer_column].astype(str).tolist()
            else:
                answers = ["" for _ in range(len(self.prompts))]
        return answers

    def get_systems(self, cfg, df):
        if cfg.dataset.system_column != "None":
            if cfg.dataset.system_column not in df.columns:
                logger.warning(
                    f"System column '{cfg.dataset.system_column}' not found."
                    " Disabling functionality."
                )
                systems = ["" for _ in range(len(self.prompts))]
            else:
                systems = df[cfg.dataset.system_column].astype(str).tolist()
        else:
            systems = ["" for _ in range(len(self.prompts))]
        return systems

    @staticmethod
    def get_conversation_ids(id2parent_id, end_id):
        """
        Gets the conversation chain for a given starting conversation ID.
        Args:
            id2parent_id: A dictionary containing the mapping of IDs
            to its previous parent ID.
            end_id: The ID of the end of the conversation in the chain.
        Returns:
            A list of conversation IDs representing the conversation chain.
            The chain is ordered from the first conversation id to end_id in the chain.
        """
        # prevent infinite loops in case
        # of circular parent chains (dataframe issue)
        loop_counter = 0

        conversation_chain_ids = [end_id]
        parent_id = end_id
        while parent_id in id2parent_id:
            loop_counter += 1

            parent_id = id2parent_id[parent_id]
            conversation_chain_ids = [parent_id] + conversation_chain_ids
            if loop_counter > 1000:
                raise ValueError(
                    f"Parent chain of sample with idx {end_id} "
                    f"exceeds max loop count of 1000. "
                    f"Please ensure that parent chain is not circular."
                )
        return conversation_chain_ids

    def __len__(self):
        return len(self.conversation_chain_ids)

    def __getitem__(self, idx):
        """
        Gets a single conversation chain.
        The conversation may be:
        - a single (system, prompt, answer) round,
          if cfg.dataset.parent_id_column == "None" or
          there is no parent_id for the conversation
        - a conversation potentially starting somewhere in
          the middle of the conversation, if the conversation
          is chained and limit_chained_samples is set to False
        - always a complete conversation, if the conversation is chained
          and limit_chained_samples is True

        """
        prompts = [self.prompts[i] for i in self.conversation_chain_ids[idx]]
        answers = [self.answers[i] for i in self.conversation_chain_ids[idx]]
        systems = [self.systems[i] for i in self.conversation_chain_ids[idx]]
        return {
            "prompts": prompts,
            "answers": answers,
            "systems": systems,
        }

    def get_conversation_end_ids(self):
        """
        Gets the end conversation IDs for each conversation chain.
        """
        return [
            conversation_chain[-1] for conversation_chain in self.conversation_chain_ids
        ]


def get_conversation_chains(
    df, cfg, limit_chained_samples=True
) -> List[Dict[str, List[str]]]:
    with PatchedAttribute(cfg.dataset, "limit_chained_samples", limit_chained_samples):
        conversation_chain_handler = ConversationChainHandler(df, cfg)
    conversations = [
        conversation
        for conversation in conversation_chain_handler  # type: ignore[attr-defined]
    ]
    return conversations
