import email
from glob import glob
import os
from utils.util import enpickle

__author__ = 'kensk8er'


def convert_eml_txt(file_name):
    message = email.message_from_file(file_name)
    #if message.is_multipart():
    #    text = message.get_payload()[0].get_payload()
    #else:
    #    text = message.get_payload()
    while message.is_multipart():
        message = message.get_payload()[0]

    text = message.get_payload()


    return text


def read_eml(directory_path):
    return_dict = {}
    file_names = glob(directory_path + '/' + '*.eml')
    count = 0
    file_num = len(file_names)

    for FILE in file_names:
        count += 1
        print '\r', count, '/', file_num,
        dir_name, file_name = os.path.split(FILE)
        file_name = file_name.split('.')[0]
        eml_file = open(FILE, 'r')
        text = convert_eml_txt(eml_file)
        return_dict[file_name] = {}
        return_dict[file_name]['text'] = text

    return return_dict


if __name__ == '__main__':
    print 'reading eml files and converting them into text data...'
    documents = read_eml('data/eml')

    print 'save them into .pkl file...'
    enpickle(documents, 'data/txt/documents.pkl')
