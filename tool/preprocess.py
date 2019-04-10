"""
FILENAME:       PREPROCESS
DESCRIPTION:    preprocess tools for captions and images
"""
import json
import logging
import h5py
import numpy as np
import os
from PIL import Image
import skimage.io
import torch
from torchvision import transforms
import nltk

import utils.netcore as netcore
import utils.resnet as resnet


def preprocess_captions(opts):
    # load file path
    path_to_input_json = opts.input_caption_json
    path_to_output_json = opts.output_caption_json
    path_to_output_h5 = opts.output_caption_h5
    image_root = opts.image_root

    assert path_to_input_json != "", "Path to input caption json is needed."
    assert path_to_output_json != "", "Path to output caption json is needed."
    assert path_to_output_h5 != "", "Path to output caption h5 is needed."

    logging.info("Path to input caption json: %s" % path_to_input_json)
    logging.info("Path to output caption json: %s" % path_to_output_json)
    logging.info("Path to output caption h5: %s" % path_to_output_h5)
    logging.info("Image root: %s" % image_root)

    # load images and dataset
    inputs = json.load(open(path_to_input_json, 'r'))
    images = inputs["images"]
    num_images = len(images)
    num_captions = sum(len(image["sentences"]) for image in images)
    dataset = inputs["dataset"]
    logging.info("Processing dataset: %s" % dataset)
    logging.info("Number of images loaded: %d" % num_images)
    logging.info("Number of captions loaded: %d" % num_captions)

    # load word threshold and maximal sentence length
    word_threshold = opts.word_threshold
    max_sentence_length = opts.max_sentence_length
    logging.info("Word occurrences threshold: %d" % word_threshold)
    logging.info("Maximal sentence length: %d" % max_sentence_length)

    # count word occurrences, noun occurrences and sentence length
    logging.info("Counting occurrences of words and nouns")
    word_occurrences = dict()
    word_count = 0
    sentence_lengths = dict()
    noun_occurrences = dict()
    noun_count = 0
    for count, image in enumerate(images, start=1):
        for sentence in image["sentences"]:
            length = len(sentence["tokens"])
            sentence_lengths[length] = sentence_lengths.get(length, 0) + 1
            # logging.debug(sentence["tokens"])
            tags = nltk.pos_tag(sentence["tokens"])
            # logging.debug(tags)
            for tag in tags:
                word_occurrences[tag[0]] = word_occurrences.get(tag[0], 0) + 1
                if word_occurrences[tag[0]] == 1:
                    word_count += 1
                if tag[1].startswith('N'):
                    noun_occurrences[tag[0]] = noun_occurrences.get(tag[0], 0) + 1
                    if noun_occurrences[tag[0]] == 1:
                        noun_count += 1
        if count % 1000 == 0:
            logging.info("%d images complete" % count)

    # sort it! big first!
    ordered_words = sorted([(times, word) for word, times in word_occurrences.items()], reverse=True)
    logging.debug("Top 10 common words:\n" +
                  "\n".join(map(str, ordered_words[:10])))
    ordered_nouns = sorted([(times, noun) for noun, times in noun_occurrences.items()], reverse=True)
    logging.debug("Top 10 common nouns:\n" +
                  "\n".join((map(str, ordered_nouns[:10]))))

    # statistics about occurrences
    word_sum = sum(word_occurrences.values())
    vocabulary = list()  # all legal words
    noun_sum = sum(noun_occurrences.values())
    nouns = list()  # all legal nouns
    rares = list()  # all illegal words
    for word, times in word_occurrences.items():
        if times <= word_threshold:
            rares.append(word)
        else:
            if word in noun_occurrences.keys():
                nouns.append(word)
            vocabulary.append(word)

    rare_sum = sum(word_occurrences[rare] for rare in rares)
    logging.info("Size of vocabulary: %d / %d (%.2f%%)" %
                 (len(vocabulary), len(word_occurrences), len(vocabulary) * 100.0 / len(word_occurrences)))
    logging.info("Size of nouns: %d / %d (%.2f%%)" %
                 (len(nouns), len(word_occurrences), len(nouns) * 100.0 / len(word_occurrences)))
    logging.info("Number of nouns: %d / %d (%.2f%%)" %
                 (noun_sum, word_sum, noun_sum * 100.0 / word_sum))
    logging.info("Size of rares: %d / %d (%.2f%%)" %
                 (len(rares), len(word_occurrences), len(rares) * 100.0 / len(word_occurrences)))
    logging.info("Number of rares, or UNK replacements: %d / %d (%.2f%%)" %
                 (rare_sum, word_sum, rare_sum * 100.0 / word_sum))

    # statistics about sentences length
    sentence_max_length = max(sentence_lengths.keys())
    sentence_sum = sum(sentence_lengths.values())
    logging.info("Maximal sentence length: %d" % sentence_max_length)
    logging.debug("Distribution of sentence lengths (length | number | ratio):")
    for i in range(sentence_max_length + 1):
        logging.debug("%2d | %7d | %2.5f%%" %
                      (i, sentence_lengths.get(i, 0), sentence_lengths.get(i, 0) * 100.0 / sentence_sum))

    # insect the token UNK,
    if rare_sum > 0:
        logging.info("Threshold detected, inserting the token UNK")
        vocabulary.append("UNK")
        nouns.append("UNK")
        noun_occurrences["UNK"] = 0

    # NOTE::create mapping between index and word, as well as between index and noun, 1-indexed!!!
    dict_index_to_word = dict()
    dict_word_to_index = dict()

    dict_noun_to_index = dict()
    array_nouns_indices = list()

    for index, word in enumerate(vocabulary, start=1):
        dict_index_to_word[index] = word
        dict_word_to_index[word] = index
        if word in nouns:
            array_nouns_indices.append(index)

    # NOTE::index for noun is independent
    for index, noun in enumerate(nouns, start=1):
        dict_noun_to_index[noun] = index

    # NOTE: encode all captions into a large array for h5 storage, 1-indexed!!!
    logging.info("Encoding all captions into one array")
    array_captions = list()  # indexed captions
    array_index_start = np.zeros(num_images, dtype='uint32')  # start index of the image
    array_index_end = np.zeros(num_images, dtype='uint32')  # end index of the image
    array_lengths = np.zeros(num_captions, dtype='uint32')  # lengths of captions

    # nouns specially
    dict_nouns = dict()  # nouns for captions
    dict_nouns_captions = dict()  # nouns for each caption

    caption_count = 0  # for all captions, 0-indexed
    caption_per_image_start = 1
    for index, image in enumerate(images):
        sentences_per_image_num = len(image["sentences"])
        assert sentences_per_image_num > 0, "No caption for this image???"

        captions_per_image = np.zeros((sentences_per_image_num, max_sentence_length), dtype='uint32')

        for tag, sentence in enumerate(image["sentences"]):
            array_lengths[caption_count] = min(max_sentence_length, len(sentence))
            caption_count += 1

            for pos, word in enumerate(sentence["tokens"]):
                # trunk to max_length
                if pos < max_sentence_length:
                    if word not in rares:
                        captions_per_image[tag, pos] = dict_word_to_index[word]
                        if word in nouns:
                            # for every noun, store the caption index and position (noun index to caption index & pos)
                            dict_nouns[dict_noun_to_index[word]] = dict_nouns.get(dict_noun_to_index[word], [])
                            dict_nouns[dict_noun_to_index[word]].append((caption_per_image_start + tag, pos))
                            # for every caption, store the noun index and position (caption index to noun index & pos)
                            dict_nouns_captions[caption_per_image_start + tag] = dict_nouns_captions.get(caption_per_image_start + tag, [])
                            dict_nouns_captions[caption_per_image_start + tag].append((dict_noun_to_index[word], pos))
                    else:
                        captions_per_image[tag, pos] = dict_word_to_index["UNK"]
                        dict_nouns["UNK"] = dict_nouns.get("UNK", [])
                        dict_nouns["UNK"].append((caption_per_image_start + tag, pos))
                        dict_nouns_captions[caption_per_image_start + tag] = dict_nouns_captions.get(caption_per_image_start + tag, [])
                        dict_nouns_captions[caption_per_image_start + tag].append((dict_noun_to_index["UNK"], pos))

        array_captions.append(captions_per_image)
        array_index_start[index] = caption_per_image_start
        array_index_end[index] = caption_per_image_start + sentences_per_image_num - 1
        caption_per_image_start += sentences_per_image_num

        if (index + 1) % 1000 == 0:
            logging.info("%d images complete" % (index + 1))

    # concatenate together
    all_captions = np.concatenate(array_captions, axis=0)
    logging.info("Size of the captions array: " + str(all_captions.shape))
    assert all_captions.shape[0] == num_captions, "Numbers are not matched, something is wrong???"
    assert np.all(array_lengths > 0), "Some captions have no words???"
    logging.info("Encode all captions into one array complete")

    # create the h5 file, not including the new nouns structures
    logging.info("Creating h5 file: %s" % path_to_output_h5)
    output_h5 = h5py.File(path_to_output_h5, 'w')
    logging.info("Writing encoded captions")
    output_h5.create_dataset("captions", dtype='uint32', data=all_captions)
    logging.info("Writing start index for every image in array captions")
    output_h5.create_dataset("index_start", dtype='uint32', data=array_index_start)
    logging.info("Writing end index for every image in array captions")
    output_h5.create_dataset("index_end", dtype='uint32', data=array_index_end)
    logging.info("Writing lengths for every caption")
    output_h5.create_dataset("caption_lengths", dtype='uint32', data=array_lengths)
    output_h5.close()
    logging.info("Create h5 file complete")

    # create the json file
    logging.info("Creating json file: %s" % path_to_output_json)
    output_json = dict()
    logging.info("Writing word index")
    output_json["index_to_word"] = dict_index_to_word
    logging.info("Writing indices of nouns")
    output_json["nouns_indices"] = array_nouns_indices
    logging.info("Writing noun index")
    output_json["noun_to_index"] = dict_noun_to_index
    logging.info("Writing nouns in captions, each entry has a list of captions and corresponding position")
    output_json["nouns_in_captions"] = dict_nouns
    logging.info("Writing captions for nouns , each entry has a list of nouns and corresponding position")
    output_json["captions_for_nouns"] = dict_nouns_captions

    logging.info("Writing image info ")
    output_json["images"] = list()
    if image_root == "":
        logging.warning("No image root specified, width and height will not be stored")

    for index, image in enumerate(images):
        output_image = dict()
        output_image["split"] = image["split"]
        output_image["filepath"] = os.path.join(image["filepath"], image["filename"])
        output_image["cocoid"] = image["cocoid"]
        if image_root != "":
            with Image.open(os.path.join(image_root, output_image["filepath"])) as img:
                output_image["width"], output_image["height"] = img.size

        output_json["images"].append(output_image)

    json.dump(output_json, open(path_to_output_json, 'w'))
    logging.info("Create json file complete")
    logging.info("Preprocess for captions complete")


def preprocess_features(opts, device):
    # load file path
    path_to_input_json = opts.input_caption_json
    directory_of_output = opts.output_feature_directory
    image_root = opts.image_root
    path_to_models = opts.model_directory

    assert path_to_input_json != "", "Path to input feature json is needed."
    assert directory_of_output != "", "Directory of output is needed."
    assert image_root != "", "Image Root is needed."
    assert path_to_models != "", "Path to models is needed."

    logging.info("Path to input feature json: %s" % path_to_input_json)
    logging.info("Directory of output: %s" % directory_of_output)
    logging.info("Image Root: %s" % image_root)
    logging.info("Path to models: %s" % path_to_models)

    # model for extraction
    model_name = opts.model
    attention_size = opts.attention_size

    logging.info("Model: %s" % model_name)
    logging.info("Attention Feature size: %d" % attention_size)

    # normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # model, use the pretrained weights
    logging.info("Loading pretrained resnet model")
    # if model_name == "resnet50":
    #     model = models.resnet50()
    # elif model_name == "resnet101":
    #     model = models.resnet101()
    # else:
    #     model = models.resnet152()

    model = getattr(resnet, model_name)()

    model.load_state_dict(torch.load(os.path.join(path_to_models, model_name + ".pth")))
    logging.debug("Current Model: \n" + model.__str__())

    feature_net = netcore.my_resnet(model)
    feature_net.to(device=device)
    feature_net.eval()
    logging.info("Load pretrained resnet model complete")

    images = json.load(open(path_to_input_json, 'r'))["images"]
    num_images = len(images)

    # feature directories
    logging.info("Creating h5 files")
    file_of_fc_feature = h5py.File(os.path.join(directory_of_output, "feats_fc.h5"))
    file_of_att_feature = h5py.File(os.path.join(directory_of_output, "feats_att.h5"))

    # feature extraction
    logging.info("Extracting features")
    for index, image in enumerate(images):
        input_image = skimage.io.imread(os.path.join(image_root, image["filepath"], image["filename"]))
        # gray_scale images
        if len(input_image.shape) == 2:
            input_image = input_image[:, :, np.newaxis]  # add one dimension
            input_image = np.concatenate((input_image, input_image, input_image), axis=2)

        input_image = input_image.astype('float32') / 255.0
        input_img = torch.from_numpy(input_image.transpose([2, 0, 1])).to(device=device)
        input_img = normalize(input_img).to(device=device)

        # extract features
        with torch.no_grad():
            feat_fc, feat_att = feature_net(input_img, attention_size)
            logging.debug("%s %s" % (feat_fc.shape, feat_att.shape))

        file_of_fc_feature.create_dataset(str(image["cocoid"]),
                                          dtype="float32",
                                          data=feat_fc.to("cpu", torch.float).numpy())
        file_of_att_feature.create_dataset(str(image["cocoid"]),
                                           dtype="float32",
                                           data=feat_att.to("cpu", torch.float).numpy())

        if index % 100 == 0:
            logging.info('Processing %d / %d (%.2f%%)' % (index, num_images, index * 100.0 / num_images))

    logging.info("Extraction complete")

    file_of_fc_feature.close()
    file_of_att_feature.close()
    logging.info("Create h5 files complete")
