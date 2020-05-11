import unittest
import context
from gantools import cli

def validate_args(args):
    # check if username, password, and keys are all set or all empty
    if not (lambda l: (not any(l)) or all(l))(\
            [e is not None and e is not [] for e in [args.username, args.password, args.keys]]):
        print(args.username)
        print(args.password)
        print(args.keys)
        raise Exception('ganbreeder login credentials are invalid')

class TestCli(unittest.TestCase):
    def test_handle_args_ganbreeder_login_correct(self):
        argv = [
                '-u', 'test@email.com',# user name
                '-p', 'password123',# password
                '-k', 'aaaa', 'bbbb', 'cccc',# keys
                ]
        args = cli.handle_args(argv=argv)
        validate_args(args)

    def test_handle_args_ganbreeder_login_no_user(self):
        argv = [
                '-p', 'password123',# password
                '-k', 'aaaa', 'bbbb', 'cccc',# keys
                ]
        try:
            args = cli.handle_args(argv=argv)
        except:
            return # this is supposed to cause an exception
        validate_args(args)

    def test_handle_args_ganbreeder_login_no_pass(self):
        argv = [
                '-u', 'test@email.com',# user name
                '-k', 'aaaa', 'bbbb', 'cccc',# keys
                ]
        try:
            args = cli.handle_args(argv=argv)
        except:
            return # this is supposed to cause an exception
        validate_args(args)

    def test_handle_args_ganbreeder_login_no_keys(self):
        argv = [
                '-u', 'test@email.com',# user name
                '-p', 'password123',# password
                ]
        try:
            args = cli.handle_args(argv=argv)
        except:
            return # this is supposed to cause an exception
        validate_args(args)

    def test_handle_args_ganbreeder_login_empty(self):
        argv = []
        args = cli.handle_args(argv=argv)
        validate_args(args)

if __name__ == '__main__':
    unittest.main()
