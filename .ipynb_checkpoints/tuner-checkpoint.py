import torch, copy, random
import matplotlib.pylot as plt

class Tuner:
    def __init__(self, config, trainer):
        '''
        config is a dictionary of (hparam:values) pairs
        that need to be tuned.

        model/model2 is the model class that 
        hasn't been instantiated yet.
        '''

        self.config = config
        self.trainer = trainer

    def set_other_hparams(self, temp_config):
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

        all_histories = {}
        for hparam, values in self.config.items():
            print ("Currently tuning : ", hparam)
            config_cp.pop(hparam)

            best_acc = float('-inf')
            best_choice = None

            if best_choice == None:
                current_hparams = self.set_other_hparams(config_cp)
            else:
                current_hparams = self.set_other_hparams(config_cp)

                # set the existing best hparams into the new config
                for done_hparam, val in final_set.items():
                    current_hparams[done_hparam] = val

            hparam_history = []

            for choice in values:
                current_hparams[hparam] = choice
                print ("Complete list: ", current_hparams)

                val_acc, avg_loss, history = self.trainer.fit(current_hparams, verbose=False, return_history=True)

                hparam_history[choice] = {
                    "history": history,
                    "hparam_setup": current_hparams
                }

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_choice = choice

            final_set[hparam] = best_choice
            all_histories[hparam] = hparam_history

        return final_set, all_histories

    def ablate_hparams_val_accs(self, all_histories):
        for hparam, hparam_history in all_histories.items():
            for choice, metadata in hparam_history.items():
                training_history = metadata['history']
                plt.plot(metadata['epochs'], metadata['val_acc'], label=repr(choice))

            plt.legend()
            plt.title("Sensitivity of {} on Validation Accuracy".format(hparam))
            plt.show()

    def ablate_hparam_train_losses(self, all_histories):
        for hparam, hparam_history in all_histories.items():
            for choice, metadata in hparam_history.items():
                training_history = metadata['history']
                plt.plot(metadata['epochs'], metadata['train_loss'], label=repr(choice))

            plt.legend()
            plt.title("Sensitivity of {} on Validation Accuracy".format(hparam))
            plt.show()
