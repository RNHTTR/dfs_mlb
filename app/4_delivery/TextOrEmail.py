'''
Placeholder file for delivering team results
'''
import sys

import pandas as pd

sys.path.append('../..')
from utils.ReadConfig import read_config


def main(sid, auth_token, to_number, from_number, input_file_name):
    ##################### EMAIL #####################
    # Import smtplib for the actual sending function
    # import smtplib
    #
    # # Here are the email package modules we'll need
    # from email.mime.image import MIMEImage
    # from email.mime.multipart import MIMEMultipart

    # Do some stuff
    #################################################

    ##################### TEXT ######################
    from twilio.rest import Client

    df = pd.read_csv(input_file_name, index_col=0)
    # NOTE: Twilio trial texts include annoying line at beginning and makes the
    #       table difficult to read without a line break.
    body = "-\n{}".format(df.to_string())

    client = Client(sid, auth_token)

    message = client.messages.create(
        to=to_number,
        from_=from_number,
        body=body)
    #################################################

if __name__ == '__main__':
    config = read_config('../config.yaml')['4_delivery']['Twilio']

    sid             = config['sid']
    auth_token      = config['auth_token']
    to_number       = config['to_number']
    from_number     = config['from_number']
    input_file_name = config['input_file_name']

    main(sid, auth_token, to_number, from_number, input_file_name)
