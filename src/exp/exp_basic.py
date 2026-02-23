from src.models import AutoTimes_Gpt2, AutoTimes_Gpt2_concatanate
from src.models import DeepTPC, DeepTPC_mix, DeepTPC_orig


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'AutoTimes_Gpt2': AutoTimes_Gpt2,
            'AutoTimes_Gpt2_concatanate': AutoTimes_Gpt2_concatanate,
            'GPT2WithMM': DeepTPC,
            'GPT2WithMM2': DeepTPC_mix,
            'GPT2WithMMWithPrompt': DeepTPC_orig,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
