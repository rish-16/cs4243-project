import torch, copy, random

class GreedySearch:
    def __init__(self, config, trainer, model):
        '''
        config is a dictionary of (hparam:values) pairs
        that need to be tuned.

        model/model2 is the model class that 
        hasn't been instantiated yet.
        '''

        self.config = config
        self.model = model
        self.trainer = trainer

    def _sample_rest(self, temp_config):
        '''
        Takes in the config and randomly choses 
        the other hparams to be picked.
        '''

        others = {}
        for hparam, values in temp_config.items():
            others[hparam] = values[0]

        return others

    def tune(self):
        config_cp = copy.deepcopy(self.config)

        final_set = {}
        for hparam, _ in config_cp.items():
            final_set[hparam] = None

        for hparam, values in self.config.items():
            print ("Currently tuning : ", hparam)
            config_cp.pop(hparam)

            best_acc = float('-inf')
            best_choice = None

            if best_choice == None:
                other_hparams = self._sample_rest(config_cp)
            else:
                other_hparams = self._sample_rest(config_cp)
                for done_hparam, val in final_set.items():
                    other_hparams[done_hparam] = val

            for choice in values:
                other_hparams[hparam] = choice
                print ("Complete list: ", other_hparams)

                val_acc, avg_loss = self.trainer.fit(other_hparams, self.model)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_choice = choice

            final_set[hparam] = best_choice

        return final_set

# def greedy_test():
#     config = {
#         "channel_list": [
#             [64, 192, 256]
#         ],
#         "dropout": [0.3],
#         "hidden_dim": [256],
#         "pool_option": [(1,1)],
#         "learning_rate": [0.02]
#     }

#     engine = GreedySearch(config, V2ConvNet)
#     optimal_set = engine.tune()

#     return optimal_set

# print (greedy_test())
