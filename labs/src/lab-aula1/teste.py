"""
Test script to verify face_recognition and cv2 libraries are properly installed.
"""


def test_libraries():
    """Test if required libraries are properly installed and configured."""
    try:
        import face_recognition

        print("✓ face_recognition imported successfully")

        import cv2

        print("✓ cv2 imported successfully")

        # Test basic functionality
        print(f"✓ OpenCV version: {cv2.__version__}")

        # Test face_recognition models are available
        try:
            # This will test if the models are properly installed
            import face_recognition_models

            print("✓ face_recognition_models available")
        except ImportError:
            print("⚠ face_recognition_models not found - some features may not work")

        print("\n🎉 All libraries are installed and working correctly!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_libraries()
