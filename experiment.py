'''
This example code shows how to conduct adversarial attacks against a code clone detection model
'''
import OpenAttack
import transformers
import datasets
import ssl
import argparse

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
    
def attack(model_path: str, attacker: OpenAttack.attackers.Attacker):
    tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, output_hidden_states=False)
    victim = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    victim = CodeCloneWrapper(victim)
    
    ## TODO: attacker accutomization
    attacker = OpenAttack.attackers.PWWSAttacker()
    
    # Load code clone detection dataset
    dataset = datasets.load_dataset("code_x_glue_cc_clone_detection_big_clone_bench", split="test[:20]").map(function=dataset_mapping)
    
    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate(),
        OpenAttack.metric.Levenshtein()
    ])
    attack_eval.eval(dataset, visualize=True)

def main(model_path: str):
    print("Start PWWS Attack")
    attack(model_path, OpenAttack.attackers.PWWSAttacker())

    print("Start TextFooler Attack")
    attack(model_path, OpenAttack.attackers.TextFoolerAttacker())

    print("Start BERT Attack")
    attack(model_path, OpenAttack.attackers.BERTAttacker())

    print("Start BAE Attack")
    attack(model_path, OpenAttack.attackers.BAEAttacker())
    
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="microsoft/codebert-base")
    args = args.parse_args()
    main(args.model_path)