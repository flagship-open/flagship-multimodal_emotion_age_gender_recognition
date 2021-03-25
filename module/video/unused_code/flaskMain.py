from flask import Flask
from flask.views import MethodView
from TestAPI import testAPIInner
from os import path


class API(MethodView):
    def __init__(self):
        pass

    def get(self, seq_path):
        """
        flask main communicator
        input as a path of the sequence folder to calculate the probability of emotion, gender and the age.

        :param seq_path: path of the sequence folder
        :return:
                valid input: json output of probability of emotion, gender and age
                invalid input: txt msg
        """

        if seq_path == 'test':
            seq_folder_path = 'seqA'
        else:
            seq_folder_path = seq_path

        seq_folder_abs_path = path.join('samples', seq_folder_path)

        if not path.exists(seq_folder_abs_path):
            return 'No seqPath exists; ' + seq_path
        else:
            return testAPIInner(seq_folder_abs_path)


class Server:
    def __init__(self):
        app = Flask(__name__)
        api_instance = API()
        app.add_url_rule('/seqPath/<string:seqPath>', view_func=api_instance.as_view('wrapper'), methods=['GET', ])

        # Run
        app.run(host='127.0.0.1', port=6009, threaded=False, debug=False)


if __name__ == '__main__':
    api = Server()
