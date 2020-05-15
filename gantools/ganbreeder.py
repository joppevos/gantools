# client functions for interacting with the ganbreeder api
import requests
import json
import numpy as np

def login(username, password):
    def get_sid():
        url = 'https://artbreeder.com/login'
        r = requests.get(url)
        r.raise_for_status()
        for c in r.cookies:
            if c.name == 'connect.sid': # find the right cookie
                print('Session ID: ' + str(c.value))
                return c.value

    def login_auth(sid, username, password):
        url = 'https://artbreeder.com/login'
        headers = {
                'Content-Type': 'application/json',
                }
        cookies = {
                'connect.sid': sid
                }
        payload = {
                'email': username,
                'password': password
                }
        r = requests.post(url, headers=headers, cookies=cookies, data=json.dumps(payload))
        if not r.ok:
            print('Authentication failed')
            r.raise_for_status()
        print('Authenticated')

    sid = get_sid()
    login_auth(sid, username, password)
    return sid

def parse_info_dict(info):
    keyframe = dict()
    keyframe['truncation'] = np.float(info['truncation'])
    keyframe['latent'] = np.asarray(info['latent'])
    classes = info['classes']
    keyframe['label'] = np.zeros(1000)# length of label ("classes") vector: 1000
    for c in info['classes']:
        # artbreeder class entries look like [index, value] where index < 1000
        keyframe['label'][c[0]] = c[1]
    return keyframe

def get_info(sid, key):
    if sid == '':
        raise Exception('Cannot get info; session ID not defined. Be sure to login() first.')
    cookies = {
            'connect.sid': sid
            }
    r = requests.get('http://artbreeder.com/info?k='+str(key), cookies=cookies)
    r.raise_for_status()
    return parse_info_dict(r.json())

def get_info_batch(username, password, keys):
    l = list()
    sid = login(username, password)
    for key in keys:
        print(key)
        l.append(get_info(sid, key))
    return l
