"""Verify bounded ZIP range extraction without accessing the network."""
import io
import struct
import unittest
from unittest.mock import patch
import zipfile

from fetch_fiord_original import RangeFile, camera_records


class Response(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range}


class DownloadTests(unittest.TestCase):
    def test_read_small_member_without_downloading_archive(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr('calibration', b'camera data')
            z.writestr('big_unused_member', b'x' * 300000)
        blob = buf.getvalue()
        transferred = []
        def open_range(req, timeout):
            requested = req.get_header('Range').split('=')[1]
            if requested.startswith('-'):
                start, end = max(0, len(blob) - int(requested[1:])), len(blob) - 1
            else:
                start, end = map(int, requested.split('-'))
            chunk = blob[start:end + 1]
            transferred.append(len(chunk))
            return Response(chunk, 206, f'bytes {start}-{end}/{len(blob)}')
        with patch('urllib.request.urlopen', side_effect=open_range):
            remote = RangeFile('https://example.invalid/archive.zip')
            with zipfile.ZipFile(remote) as z:
                self.assertEqual(z.read('calibration'), b'camera data')
            self.assertLess(sum(transferred), len(blob) / 2)
            self.assertEqual(remote.received, sum(transferred))

    def test_server_ignoring_range_is_rejected(self):
        with patch('urllib.request.urlopen', return_value=Response(b'not a range', 200, '')):
            with self.assertRaisesRegex(ValueError, 'honor Range'):
                RangeFile('https://example.invalid/archive.zip')

    def test_budget_prevents_request(self):
        with patch('urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'budget'):
                RangeFile('https://example.invalid/archive.zip', budget=100)
            request.assert_not_called()

    def test_camera_id_and_params(self):
        params = [100, 101, 200, 201, .01, .02, .03, .04]
        blob = struct.pack('<QiiQQ8d', 1, 7, 5, 400, 401, *params)
        cameras = camera_records(blob)
        self.assertEqual(cameras[7]['params'], params)
        self.assertEqual(cameras[7]['width'], 400)
        self.assertEqual(cameras[7]['height'], 401)


if __name__ == '__main__':
    unittest.main()
