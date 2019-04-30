"""
FILENAME:       OPTIONS
DESCRIPTION:    options for the whole program, a mistake I made
"""

import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description="SCT PROTO V1 #ELKA")

    # core mode
    parser.add_argument('-m', "--mode",
                        dest="mode",
                        default="train",
                        required=True,
                        choices=["train", "eval", "precaps", "prefeats"],
                        help="Choose the mode")

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help="Debug mode")

    # cuda device
    parser.add_argument('-cuda', "--cuda_device",
                        type=str,
                        default='',
                        help="The cuda device")

    # preprocess captions
    parser.add_argument('-incapjson', "--input_caption_json",
                        default="",
                        help="Path to input split json to preprocess the captions")

    parser.add_argument('-outcapjson', "--output_caption_json",
                        default="data/sct_caps2idx.json",
                        help="Path to output the result json of processing the captions")

    parser.add_argument('-outcaph5', "--output_caption_h5",
                        default="data/sct_caps.h5",
                        help="Path to output the result h5 of processing the captions")

    parser.add_argument('-maxlen', "--max_sentence_length",
                        default=16,
                        help="The maximal length of sentence, exceeding part will be truncated")

    parser.add_argument('-wordthres', "--word_threshold",
                        default=5,
                        help="The threshold of the number of occurrences of words in all captions, "
                             "rare words will be replaced by token UNK")
    parser.add_argument('-imgrt', "--image_root",
                        default="",
                        help="root directory of images")

    # preprocess images, including image_root and input_caption_json
    parser.add_argument('-outfeatdir', "--output_feature_directory",
                        default='data/features',
                        help="directory for output feature files")

    parser.add_argument('-attsize', "--attention_size",
                        default=14,
                        choices=[7, 14],
                        type=int,
                        help="attention size")

    parser.add_argument('-mod', "--model",
                        default="resnet101",
                        choices=["resnet50", "resnet101", "resnet152"],
                        type=str,
                        help="model for feature extraction")

    parser.add_argument('-moddir', "--model_directory",
                        default="data/models",
                        type=str,
                        help="directory of the models for feature extraction")

    # training
    parser.add_argument('-id', "--train_id",
                        type=str,
                        default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress '
                             'files')

    parser.add_argument('-injson', "--input_json",
                        type=str,
                        default="data/sct_caps2idx.json",
                        help="path to the input json containing the image info and vocabulary")

    parser.add_argument('-infeatdir', "--input_features_directory",
                        type=str,
                        default="",
                        help="directory of h5 files containing fc features and att features")

    parser.add_argument('-incaph5', "--input_captions_h5",
                        type=str,
                        default="data/sct_caps.h5",
                        help="path to the input h5 containing the indexed captions and index")

    parser.add_argument('-outmemdir', "--output_memory_directory",
                        type=str,
                        default="data/memory",
                        help="directory for the output memory file")

    parser.add_argument('-inmemh5', '--input_memory_h5',
                        type=str,
                        default='',
                        help='h5 file for input memory if skipping to stage 3')

    parser.add_argument('-batch', "--batch_size",
                        default=64,
                        type=int,
                        help="size of one mini-batch")

    parser.add_argument('-batch3', '--batch_size_3',
                        type=int,
                        default= 16,
                        help="size of one mini-batch for stage 3")

    parser.add_argument('-capsperimg', "--captions_per_image",
                        default=5,
                        type=int,
                        help="number of captions to sample for each image during training")

    parser.add_argument('-beam', "--beam_size",
                        default=1,
                        type=int,
                        help="beam search size")

    parser.add_argument('-epo', "--epoch_num",
                        default=25,
                        type=int,
                        help="number of epoch")

    parser.add_argument('-epo3', '--epoch_num3',
                        default=20,
                        type=int,
                        help="number of epoch in stage 3")

    parser.add_argument('-dropout', "--dropout_prob",
                        default=0.5,
                        type=float,
                        help="probability value for dropout")

    parser.add_argument('-only', "--train_only",
                        type=int,
                        default=0,
                        help='if 1 then use 80k, else use 110k, including restval')

    parser.add_argument('-laneval', '--language_eval',
                        type=int,
                        default=0,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires '
                             'coco-caption code from Github.')

    parser.add_argument('-tm', '--train_mode',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        help='The stage of training to start from, 0 for all')

    parser.add_argument('-tdir', '--train_directory',
                        type=str,
                        default=None,
                        help="Need a skip?, tell me where to get the model and info")

    # model
    parser.add_argument('-model', '--caption_model',
                        type=str,
                        help="model we want to use")

    parser.add_argument('-rnn', '--rnn_size',
                        type=int,
                        default=512,
                        help='size of the rnn in number of hidden nodes in each layer')

    parser.add_argument('-layers', '--num_layers',
                        type=int,
                        default=1,
                        help='number of layers in the RNN')
    parser.add_argument('-type', '--rnn_type',
                        type=str,
                        default='lstm',
                        choices=['rnn', 'gru', 'lstm'],
                        help='the type of RNN')

    parser.add_argument('-inencsize', '--input_encoding_size',
                        type=int,
                        default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')

    parser.add_argument('-atthidsize', '--att_hid_size',
                        type=int,
                        default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using '
                             'hidden layer')

    parser.add_argument('-fcfeatsize', '--fc_feat_size',
                        type=int,
                        default=2048,
                        help='2048 for resnet, 4096 for vgg')

    parser.add_argument('-attfeatsize', '--att_feat_size',
                        type=int,
                        default=2048,
                        help='2048 for resnet, 512 for vgg')

    parser.add_argument('-fusecoef', '--fuse_coefficient',
                        type=float,
                        default=0.50,
                        help='fuse coeficient for output')

    parser.add_argument('-memosize', '--memory_size',
                        type=int,
                        default=512,
                        help='size of memory, same as att size (rnn_size)')

    parser.add_argument('-prefcsize', '--relation_pre_fc_size',
                        type=int,
                        default=64,
                        help='size of pre_fc in relation module')

    parser.add_argument('-postfcsize', '--relation_post_fc_size',
                        type=int,
                        default=32,
                        help='size of post_fc in relation module')

    # optimization
    parser.add_argument('-opt', '--optim',
                        type=str,
                        default='adam',
                        choices=["adam", "rmsprop", "sgd", "sgdmom", "adagrad", "adam"],
                        help='optimization type')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=4e-4,
                        help='learning rate')

    parser.add_argument('-lrdecayst', '--learning_rate_decay_start',
                        type=int,
                        default=-1,
                        help='at which iteration to start decaying learning rate (-1 = dont) (in epoch)')

    parser.add_argument('-lrdecayeve', '--learning_rate_decay_every',
                        type=int,
                        default=3,
                        help='decay start from this epoch)')

    parser.add_argument('-lrdecay', '--learning_rate_decay_rate',
                        type=float,
                        default=0.8,
                        help='learning rate decay rate')

    parser.add_argument('-alpha', '--optim_alpha',
                        type=float,
                        default=0.9,
                        help='alpha for adam')

    parser.add_argument('-beta', '--optim_beta',
                        type=float,
                        default=0.999,
                        help='beta for adam')

    parser.add_argument('-epsilon', '--optim_epsilon',
                        type=float,
                        default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    parser.add_argument('-wdecay', '--weight_decay',
                        type=float,
                        default=0,
                        help='weight_decay')

    parser.add_argument('-ssst', '--scheduled_sampling_start',
                        type=int,
                        default=-1,
                        help='at which epoch to start decay gt probability')

    parser.add_argument('-ssinevery', '--scheduled_sampling_increase_every',
                        type=int,
                        default=5,
                        help='every how many iterations thereafter to gt probability')

    parser.add_argument('-ssincp', '--scheduled_sampling_increase_prob',
                        type=float,
                        default=0.05,
                        help='value to update the prob')

    parser.add_argument('-ssmaxp', '--scheduled_sampling_max_prob',
                        type=float,
                        default=0.25,
                        help='Maximum scheduled sampling prob.')

    parser.add_argument('-gclip', '--grad_clip',
                        type=float,
                        default=0.1,
                        help='clip gradients at this value')

    parser.add_argument('-valimg', '--val_images_use',
                        type=int,
                        default=3200,
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')

    # check point
    parser.add_argument('-start', '--start_from',
                        type=str,
                        default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by 
                        previous training process: 
                            'infos.pkl'     : configuration;
                            'checkpoint'    : paths to model file(s) (created by tf).
                             Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'  : file(s) with model definition (created by tf) """)

    parser.add_argument('-chkpt', "--checkpoint_path",
                        type=str,
                        default='checkpoints',
                        help='directory to store checkpointed models')

    parser.add_argument('-best', '--load_best_score',
                        type=int,
                        default=1,
                        help='Do we load previous best score when resuming training.')

    parser.add_argument('-losslogeve', '--losses_log_every',
                        type=int,
                        default=25,
                        help='How often do we snapshot losses, for inclusion in the progress dump (0 = disable)')

    parser.add_argument('-savechkpteve', '--save_checkpoint_every',
                        type=int,
                        default=1000,
                        help='how often to save a model checkpoint (in iterations)')

    parser.add_argument('-savechkpteve3', '--save_checkpoint_every_3',
                        type=int,
                        default=3500,
                        help='how often to save a model checkpoint (in iterations) for stage 3')

    parser.add_argument('-best1', '--best_models_1',
                        type=str,
                        default='data/best1',
                        help='best models of stage 1')

    parser.add_argument('-best3', '--best_models_3',
                        type=str,
                        default='data/best3',
                        help='best models of stage 3')

    # evaluation
    parser.add_argument('-modpth', '--model_path',
                        type=str,
                        default='',
                        help="path to the model to evaluate")

    parser.add_argument('-infopth', "--info_path",
                        type=str,
                        default="",
                        help="path to info to evaluate")

    parser.add_argument('-images', "--num_images",
                        type=int,
                        default=-1,
                        help='how many images to use when periodically evaluating the loss? (-1 = all)')

    parser.add_argument('-dumpimg', '--dump_images',
                        type=int,
                        default=0,
                        help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')

    parser.add_argument('-dumpjson', '--dump_json',
                        type=int,
                        default=0,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')

    parser.add_argument('-dumppth', '--dump_path',
                        type=int,
                        default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')

    parser.add_argument('-greedy', '--sample_greedy',
                        type=int,
                        default=1,
                        help='1 = greedily decoding. 0 = sample.')

    parser.add_argument('-ppl', '--max_ppl',
                        type=int,
                        default=0,
                        help='beam search by max perplexity or max probability.')

    parser.add_argument('-group', '--group_size',
                        type=int,
                        default=1,
                        help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')

    parser.add_argument('-diver', '--diversity_lambda',
                        type=float,
                        default=0.5,
                        help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces '
                             'a more diverse list')

    parser.add_argument('-t', '--temperature',
                        type=float,
                        default=1.0,
                        help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = '
                             '"safer" predictions.')

    parser.add_argument('-decodcon', '--decoding_constraint',
                        type=int,
                        default=0,
                        help='If 1, not allowing same word in a row')

    parser.add_argument('-img', '--image_folder',
                        type=str,
                        default='',
                        help='If this is nonempty then will predict on the images in this folder path')

    parser.add_argument('-s', '--split',
                        type=str,
                        default='test',
                        help='if running on MS COCO images, which split to use: val|test|train')

    parser.add_argument('-coco', '--coco_json',
                        type=str,
                        default='',
                        help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MS COCO '
                             'test evaluation, where we have a specific json file of only test set images.')

    parser.add_argument('-evalid', "--eval_id",
                        type=str,
                        default='',
                        help='an id identifying this run/job. used only if language_eval = 1 for appending to '
                             'intermediate files')

    parser.add_argument('-verbeam', '--verbose_beam',
                        type=int,
                        default=1,
                        help='if we need to print out all beam search beams.')

    parser.add_argument('-verloss', '--verbose_loss',
                        type=int,
                        default=0,
                        help='if we need to calculate loss.')


    opts = parser.parse_args()

    return opts
