class FaceAttendanceError(Exception):
    pass


class ConfigurationError(FaceAttendanceError):
    pass


class RegistryError(FaceAttendanceError):
    pass


class AttendanceError(FaceAttendanceError):
    pass


class RecognitionError(FaceAttendanceError):
    pass


class RegistrationError(FaceAttendanceError):
    pass


class LivenessError(FaceAttendanceError):
    pass


class DependencyError(FaceAttendanceError):
    pass


class CameraError(FaceAttendanceError):
    pass
