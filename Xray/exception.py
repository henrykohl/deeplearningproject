import os
import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info() # exc type, exc value, exc tb

    file_name: str = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1] # os.path.split() 回傳 [Head, Tail]

    error_message: str = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )  # 不用 str(error) ，直接用 error 也可以

    return error_message # 新的 error_message


class XRayException(Exception):
    def __init__(self, error_message, error_detail): # error_message: Exception, error_detail: sys
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message) # error_message: Exception, 寫成 super().__init__() 似乎不影響

        self.error_message: str = error_message_detail( # 新的 error_message
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message # 新的 error_message