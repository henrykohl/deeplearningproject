import os
import sys

from Xray.exception import XRayException


class S3Operation:
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            command: str = (
                f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/ "
            )

            os.system(command)

        except Exception as e:
            raise XRayException(e, sys) # `e' is a message by default we are getting.
            # `sys' is collecting the information from the execution.

    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            command: str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} "
            ) # {folder} 之前要有空格，因為 {folder} 是目的地

            os.system(command)

        except Exception as e:
            raise XRayException(e, sys)