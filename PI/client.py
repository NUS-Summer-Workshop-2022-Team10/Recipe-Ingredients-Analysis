import requests
import json
from PIL import Image
from base64 import b64encode, b64decode

import uri

###
headers = {'Content-type': 'application/json'}


# @section: Helper functions
# This part of the library contains internal functions used later on.
#

def send_request(endpoint, request):
    """! Forms JSON POST requests to the backend.
    @param endpoint The endpoint to send to, e.g. '/user/add'
    @param request  The POST data in the form of a Python dictionary.

    @return The result from the server
    """

    return requests.post(uri.URI + endpoint, headers=headers, data=json.dumps(request))


# @section: functions be used
# look at this !!! important!!!
#

def find_data(temp=None, ir=None, wl=None):
    """! Search for data. All parameters are optional.

    @param temp  temperature be tested by sensor
    @param ir
    @param wl   water level be tested by sensor

    @returns    Data from server in this format:
    {'1': '1', '2': '1', '3': 0}
    """
    req = {}

    if temp is not None:
        req['temp'] = temp
    if ir is not None:
        req['ir'] = ir
    if wl is not None:
        req['wl'] = wl

    res = send_request('/acquire', req)

    print("Status code: ", res.status_code, " Details: ", res.json())

    return res.json()


def add_data(temp=None, ir=None, wl=None):
    """! Add new IoT sensor data to the database.

    @param temp  temperature be tested by sensor
    @param ir
    @param wl   water level be tested by sensor

    @returns    Response from server
    """

    req = {'temp': temp, 'ir': ir, 'wl': wl}
    return send_request('/add', req)


def image_classify(address, mode):
    """! Add new image to the database.

        @param address  the address of images

        @returns    Response from server
        """

    # Load the image
    image = Image.open(address)
    # Resize the image
    new_img = image.resize((480, 480))
    #new_img.show()
    # Now get the bytes
    img_bytes = new_img.tobytes()
    # And do base 64 encoding. Note we need to call "decode" to turn
    # this into an ASCII string.
    img_b64 = b64encode(img_bytes).decode('utf-8')
    # print(img_b64)
    req = {"image": img_b64, "mode":mode}
    res = send_request('/classify', req)
    print("Status code: ", res.status_code, " Details: ", res.json())
    return res.json()

def page():
    req={}
    res = send_request('/i', req)
    print("Status code: ", res.status_code, " Details: ", res.json())
    return res.json()

if __name__ == '__main__':
    page()
    #res = image_classify("./image/1-1.png", 1)
   # res1 = image_classify("./image/4.jpg", 1)
    #res2 = image_classify("./image/5.jpg", 1)

    #res3 = image_classify("./image/6.jpg", 1)
    #add_data(temp=1, ir=1, wl=1)
