class ErrorExc(Exception):
    def __init__(self, msg="Exception raised"):
        self.message = msg
        super().__init__(self.message)