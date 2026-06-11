import os
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from sana_evaluation.tools import agent_tools
from sana_evaluation.tools.external.ideal.benchmark_paths import canonical_source_uri


class TestS3PublicBucketMode(unittest.TestCase):
    def setUp(self):
        self._old_mode = agent_tools._S3_CLIENT_MODE
        self._old_signed = agent_tools._S3_SIGNED_CLIENT
        self._old_unsigned = agent_tools._S3_UNSIGNED_CLIENT
        agent_tools._S3_CLIENT_MODE = None
        agent_tools._S3_SIGNED_CLIENT = None
        agent_tools._S3_UNSIGNED_CLIENT = None

    def tearDown(self):
        agent_tools._S3_CLIENT_MODE = self._old_mode
        agent_tools._S3_SIGNED_CLIENT = self._old_signed
        agent_tools._S3_UNSIGNED_CLIENT = self._old_unsigned

    def test_auto_mode_falls_back_to_unsigned_on_access_denied(self):
        signed = Mock()
        signed.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "explicit deny"}},
            "ListObjectsV2",
        )
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "auto"}, clear=False), \
             patch.object(agent_tools, "_get_signed_s3_client", return_value=signed), \
             patch.object(agent_tools, "_get_unsigned_s3_client", return_value=unsigned):
            client = agent_tools._get_s3_client()

        self.assertIs(client, unsigned)
        self.assertEqual(agent_tools._S3_CLIENT_MODE, "unsigned")

    def test_auto_mode_uses_signed_when_probe_succeeds(self):
        signed = Mock()
        signed.list_objects_v2.return_value = {"KeyCount": 1}
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "auto"}, clear=False), \
             patch.object(agent_tools, "_get_signed_s3_client", return_value=signed), \
             patch.object(agent_tools, "_get_unsigned_s3_client", return_value=unsigned):
            client = agent_tools._get_s3_client()

        self.assertIs(client, signed)
        self.assertEqual(agent_tools._S3_CLIENT_MODE, "signed")
        unsigned.list_objects_v2.assert_not_called()

    def test_unsigned_mode_skips_signed_probe(self):
        signed = Mock()
        unsigned = Mock()

        with patch.dict(os.environ, {"S3_ACCESS_MODE": "unsigned"}, clear=False), \
             patch.object(agent_tools, "_get_signed_s3_client", return_value=signed), \
             patch.object(agent_tools, "_get_unsigned_s3_client", return_value=unsigned):
            client = agent_tools._get_s3_client()

        self.assertIs(client, unsigned)
        self.assertEqual(agent_tools._S3_CLIENT_MODE, "unsigned")
        signed.list_objects_v2.assert_not_called()

    def test_configure_benchmark_switches_to_kramabench_bucket(self):
        agent_tools.configure_benchmark("kramabench")

        self.assertEqual(agent_tools.BUCKET, "sana-kramabench")
        self.assertEqual(
            canonical_source_uri(
                "datagov/kramabench-archeology-easy-10/files/worldcities.csv",
                "kramabench",
            ),
            "s3://sana-kramabench/datagov/kramabench-archeology-easy-10/files/worldcities.csv",
        )
        self.assertEqual(
            agent_tools._parse_s3_reference(
                "s3://sana-kramabench/datagov/kramabench-archeology-easy-10/files/worldcities.csv"
            )["key"],
            "datagov/kramabench-archeology-easy-10/files/worldcities.csv",
        )

    def test_configure_benchmark_defaults_to_lakeqa_bucket(self):
        agent_tools.configure_benchmark(None)

        self.assertEqual(agent_tools.BUCKET, "lakeqa-yc4103-datalake")
        self.assertEqual(
            canonical_source_uri("datagov/example/files/rows.csv", "lakeqa"),
            "s3://lakeqa-yc4103-datalake/datagov/example/files/rows.csv",
        )


if __name__ == "__main__":
    unittest.main()
