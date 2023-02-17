import json
import os
from collections import defaultdict

from loguru import logger


class KGInformation:
    def __init__(self, args, data_path):
        super(KGInformation, self).__init__()
        self.args = args
        self.data_path = data_path
        self._load_other_data()

    def _load_other_data(self):
        # todo: KG information 분리 시켜야 함
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.data_path, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.data_path, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        self.entity_kg = self._entity_kg_process()

        self.movie2name = json.load(
            open(os.path.join(self.data_path, 'movie2name.json'), 'r', encoding='utf-8'))  # {entity: entity_id}

        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.data_path, 'entity2id.json')} and {os.path.join(self.data_path, 'dbpedia_subkg.json')}]")

        logger.debug("[Finish entity KG process]")

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
        return {
            'edge': list(edges), #[(h, t, r)]
            'n_relation': len(relation2id),
            'entity': list(entities)
        }