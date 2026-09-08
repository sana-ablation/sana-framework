import os
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from sana_evaluation.tools import lake
from sana_evaluation.tools.external.ideal.benchmark_paths import canonical_source_uri


class TestS3PublicBucketMode(unittest.TestCase):
    def setUp(self):
        self._old_mode = lake._S3_CLIENT_MODE
        self._old_signed = lake._S3_SIGNED_CLIENT
        self._old_unsigned = lake._S3_UNSIGNED_CLIENT
        lake._S3_CLIENT_MODE = None
        lake._S3_SIGNED_CLIENT = None
        lake._S3_UNSIGNED_CLIENT = None

    def tearDown(self):
        lake._S3_CLIENT_MODE = self._old_mode
        lake._S3_SIGNED_CLIENT = self._old_signed
        lake._S3_UNSIGNED_CLIENT = self._old_unsigned

    def test_auto_mode_falls_back_to_unsigned_on_access_denied(self):
        signed = Mock()
        signed.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "explicit deny"}},
            "ListObjectsV2",
        )
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "auto"}, clear=False), \
             patch.object(lake, "_get_signed_s3_client", return_value=signed), \
             patch.object(lake, "_get_unsigned_s3_client", return_value=unsigned):
            client = lake._get_s3_client()

        self.assertIs(client, unsigned)
        self.assertEqual(lake._S3_CLIENT_MODE, "unsigned")

    def test_auto_mode_uses_signed_when_probe_succeeds(self):
        signed = Mock()
        signed.list_objects_v2.return_value = {"KeyCount": 1}
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "auto"}, clear=False), \
             patch.object(lake, "_get_signed_s3_client", return_value=signed), \
             patch.object(lake, "_get_unsigned_s3_client", return_value=unsigned):
            client = lake._get_s3_client()

        self.assertIs(client, signed)
        self.assertEqual(lake._S3_CLIENT_MODE, "signed")
        unsigned.list_objects_v2.assert_not_called()

    def test_unsigned_mode_skips_signed_probe(self):
        signed = Mock()
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "unsigned"}, clear=False), \
             patch.object(lake, "_get_signed_s3_client", return_value=signed), \
             patch.object(lake, "_get_unsigned_s3_client", return_value=unsigned):
            client = lake._get_s3_client()

        self.assertIs(client, unsigned)
        self.assertEqual(lake._S3_CLIENT_MODE, "unsigned")
        signed.list_objects_v2.assert_not_called()

    def test_configure_benchmark_switches_to_kramabench_bucket(self):
        lake.configure_benchmark("kramabench")

        self.assertEqual(lake.BUCKET, "sana-kramabench")
        self.assertEqual(
            canonical_source_uri(
                "datagov/kramabench-archeology-easy-10/files/worldcities.csv",
                "kramabench",
            ),
            "s3://sana-kramabench/datagov/kramabench-archeology-easy-10/files/worldcities.csv",
        )
        self.assertEqual(
            lake._parse_s3_reference(
                "s3://sana-kramabench/datagov/kramabench-archeology-easy-10/files/worldcities.csv"
            )["key"],
            "datagov/kramabench-archeology-easy-10/files/worldcities.csv",
        )

    def test_configure_benchmark_defaults_to_lakeqa_bucket(self):
        lake.configure_benchmark(None)

        self.assertEqual(lake.BUCKET, "lakeqa-yc4103-datalake")
        self.assertEqual(
            canonical_source_uri("datagov/example/files/rows.csv", "lakeqa"),
            "s3://lakeqa-yc4103-datalake/datagov/example/files/rows.csv",
        )


if __name__ == "__main__":
    unittest.main()
