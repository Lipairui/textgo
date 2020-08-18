# coding: utf-8
import sys
from transformers import AutoTokenizer, AutoModelWithLMHead

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelWithLMHead.from_pretrained(model_name)


