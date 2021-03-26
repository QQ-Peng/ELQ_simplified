from test import *
from train import *
from mention_detection import *
import torch
import argparse
import sys

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_name",
                        default='bert-base-uncased',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model_path",
                        help="The path where to load model.")

    parser.add_argument("--train_file")
    parser.add_argument("--test_file")

    parser.add_argument("--train",
                        default=False,
                        type=bool,
                        help="whether train a new model or not.")
    parser.add_argument("--model_output_dim",
                        default=768,
                        help="the dimension of word embedding outputted by bert model.it should align with the bert model.")
    parser.add_argument("--test",
                        default=False,
                        type=bool,
                        help="whether test the model or not.")
    parser.add_argument("--learning_rate",
                        default=0.02,
                        type=float
                        )
    parser.add_argument("--mention_maxlen",
                        default=10,
                        type=int
                        )
    parser.add_argument("--score_method",
                        default='qa_mlp',
                        help="options: qa_mlp, qa_linear, qa"
                        )
    parser.add_argument("--device",
                        default='cpu',
                        )
    parser.add_argument("--batch_size",
                        default=8,
                        type=int
                        )
    parser.add_argument("--epoch",
                        default=2,
                        type=int
                        )
    parser.add_argument("--top_k",
                        default=1,
                        type=int,
                        help="set the predicted mention boundary number to evaluate, reference value: 1 or 2"
                        )
    parser.add_argument("--model_save_path",
                        default='.'
                        )

    parser.add_argument("--pre_trained_model",
                        help="a model has been trained, which can be used to do predict job.")
    # parser.add_argument("--help",
    #                     default=False,
    #                     action="store_true"
    #                     )

    params = parser.parse_args()

    if params.train:
        print('*' * 30, 'train mode', '*' * 30)
        mention_detection_model = train_model(params.bert_model_name,
                                              params.bert_model_path,
                                              params.train_file,
                                              params.batch_size,
                                              params.learning_rate,
                                              params.top_k,
                                              params.device,
                                              params.model_output_dim,
                                              params.epoch,
                                              params.score_method,
                                              params.mention_maxlen
                                              )
        torch.save(mention_detection_model.state_dict(), params.model_save_path +'/mention_det_model_rice.bin')

    if params.test:
        print('*' * 30, 'test mode', '*' * 30)

        mend_dect_model = MentionDect(params.bert_model_name,
                                      params.bert_model_path,
                                      params.model_output_dim,
                                      params.score_method,
                                      params.mention_maxlen)
        mend_dect_model.load_state_dict(torch.load(params.pre_trained_model))
        mend_dect_model = mend_dect_model.to(params.device)
        mend_dect_model.eval()
        test_model(mend_dect_model,params.bert_model_name,params.test_file,params.batch_size,params.top_k,params.device)