import unittest
from gantools import ganbreeder
from test import secrets

class TestGanbreeder(unittest.TestCase):
    def test_get_info(self):
        username = secrets.username
        password = secrets.password
        sid = ganbreeder.login(username, password)
        self.assertNotEqual(sid, '', 'login() failed to produce an sid. check internet connection.')
        key = 'd62c507ab4bea4ed7b70c64a' #some arbitrary ganbreeder key
        ganbreeder.get_info(sid, key)

if __name__ == '__main__':
    unittest.main()

