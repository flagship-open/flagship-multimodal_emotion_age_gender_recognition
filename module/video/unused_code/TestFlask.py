from urllib import request

"""
Test Flask script

Use this script or in the web browser

Flask URL format :
address + "/seqPath/" + sequence folder path in local
ex) 'http://127.0.0.1:6009/seqPath/seqB'

For Test, replace the sequence folder path with test
ex) 'http://127.0.0.1:6009/seqPath/test'

"""


base_url = 'http://127.0.0.1:6009/seqPath/'
test_url = ['test', 'seqB']

for each_test_url in test_url:
    full_url = base_url + each_test_url
    with request.urlopen(base_url + each_test_url) as response:
        html = response.read()

    print('Input: ' + full_url)
    print('Result: ')
    print(html)
    print('\n')
