'''
Placeholder file for delivering team results
'''

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

# Your Account SID from twilio.com/console
account_sid = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Your Auth Token from twilio.com/console
auth_token  = "your_auth_token"

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="+18135555555",
    from_="+15017250604",
    body="Hello from Python!")

print(message.sid)
#################################################
