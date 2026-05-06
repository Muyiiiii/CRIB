import sys
sys.path.append('../')

from TSL_models import CRIB
from TSL_models import DLinear, SegRNN, Transformer, iTransformer, PatchTST, TSMixer,WPMixer, PAttn, KANAD, MultiPatchFormer, FreTS, TimesNet # want input [Batch, seq_len, Channels]

from NeuralCDE import NeuralCDE

def select_model(args):
    if args.model=="CRIB":
        model=CRIB(args).to(args.device)
    elif args.model=="DLinear":
        model = DLinear(args).to(args.device)
    elif args.model=="SegRNN":
        model = SegRNN(args).to(args.device)
    elif args.model=="Transformer":
        model = Transformer(args).to(args.device)
    elif args.model=="iTransformer":
        model = iTransformer(args).to(args.device)
    elif args.model=="PatchTST":
        model = PatchTST(args, patch_len=args.patch_len).to(args.device)
    elif args.model=="TSMixer":
        model = TSMixer(args).to(args.device)
    elif args.model=="WPMixer":
        model = WPMixer(args).to(args.device)
    elif args.model=="PAttn":
        model = PAttn(args).to(args.device)
    elif args.model=="KANAD":
        model = KANAD(args).to(args.device)
    elif args.model=="MultiPatchFormer":
        model = MultiPatchFormer(args).to(args.device)
    elif args.model=="FreTS":
        model = FreTS(args).to(args.device)
    elif args.model=="TimesNet":
        model = TimesNet(args).to(args.device)
    elif args.model=="NeuralCDE":
        model = NeuralCDE(args).to(args.device)
    else:
        raise ValueError(f"Model-{args.model} not found")
    
    return model