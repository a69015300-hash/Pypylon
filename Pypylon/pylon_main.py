import cv2
from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv2.namedWindow("123", cv2.WINDOW_NORMAL)
camera.StartGrabbing()

try:
    while camera.IsGrabbing():
        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if grabResult.GrabSucceeded():
                    img = converter.Convert(grabResult).GetArray()
                    cv2.imshow("123", img)

                    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                        break
            finally:
                grabResult.Release()

        except pylon.TimeoutException:
            # 忽略逾時，繼續下一輪
            continue
        except pylon.GenericException:
            # 忽略 GenICam/pylon 相關錯誤示範（可視需求改成 break）
            continue
        except:
            continue
finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
