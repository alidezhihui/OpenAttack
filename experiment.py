'''
This example code shows how to conduct adversarial attacks against a code clone detection model
'''
import OpenAttack
import transformers
import datasets
import ssl
import argparse
import torch
from model import Model
ssl._create_default_https_context = ssl._create_unverified_context

class CodeCloneWrapper(OpenAttack.classifiers.Classifier):
    def __init__(self, model : OpenAttack.classifiers.Classifier):
        self.model = model
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)
    
    def get_prob(self, input_):
        func2 = self.context.input["func2"]
        input_pairs = [  func1 + " [SEP] " + func2 for func1 in input_ ]
        print(f"Input pairs: {input_pairs[:2]}...")
        return self.model.get_prob(input_pairs)

def dataset_mapping(x):
    return {
        "x": x["func1"],
        "y": x["label"],
        "func2": x["func2"]
    }
    
def attack(args: argparse.Namespace, attacker: OpenAttack.attackers.Attacker):
    # Load tokenizer from model name/path
    tokenizer = transformers.RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = transformers.RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 2
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = transformers.RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    model=Model(model,config,tokenizer,args)
    
    ob = torch.load(args.checkpoint_file_path)
    print(ob)
    model.load_state_dict(torch.load(args.checkpoint_file_path))
    model.to(args.device)
    
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    victim = CodeCloneWrapper(victim)
    
    # Load code clone detection dataset
    dataset = datasets.load_dataset("code_x_glue_cc_clone_detection_big_clone_bench", split="test[:20]").map(function=dataset_mapping)
    
    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate(),
        OpenAttack.metric.Levenshtein(tokenizer=tokenizer)
    ])
    attack_eval.eval(dataset, visualize=True)

def main(args: argparse.Namespace):
    print("Start PWWS Attack")
    attack(args, OpenAttack.attackers.PWWSAttacker())

    print("Start TextFooler Attack")
    attack(args, OpenAttack.attackers.TextFoolerAttacker())

    print("Start BERT Attack")
    attack(args, OpenAttack.attackers.BERTAttacker())

    print("Start BAE Attack")
    attack(args, OpenAttack.attackers.BAEAttacker())
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--checkpoint_file_path", type=str, default=None, required=True, help="The checkpoint file to be attacked")

    args = parser.parse_args()
    main(args)