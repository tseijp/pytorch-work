import numpy as np
import cv2

DEVICE_ID=0
cap=cv2.VideoCapture(DEVICE_ID)

# Shi-Tomasiのコーナー検出パラメータ
ftr_params=dict( maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7 )

# Lucas-Kanade法のパラメータ
lk_params=dict( winSize =(15,15),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                color=np.random.randint(0, 255, (100, 3) )

def lucas(gry1, gry2, ftr1, mask, frame):
    # オプティカルフロー検出
    ftr2, status, err=cv2.calcOpticalFlowPyrLK(gry1, gry2, ftr1, None, **lk_params)

    if len(ftr2[status == 1]) == 0:
        ftr1=cv2.goodFeaturesToTrack(gry1, mask=None, **ftr_params)
        mask=np.zeros_like(frame)
        ftr2, status, err=cv2.calcOpticalFlowPyrLK(gry1, gry2, ftr1, None, **lk_params)
    # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
    good1=ftr1[status == 1]
    good2=ftr2[status == 1]

    # オプティカルフローを描画
    for i, (next_point, prev_point) in enumerate(zip(good2, good1)):
        prev_x, prev_y=prev_point.ravel()
        next_x, next_y=next_point.ravel()
        mask=cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)
        frame=cv2.circle(frame, (next_x, next_y), 25, color[i].tolist(), -1)
    return good2.reshape(-1, 1, 2), cv2.add(frame, mask), mask

if __name__=='__main__':
    # 最初のフレームの処理
    end_flag, frame=cap.read()
    gry1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ftr1=cv2.goodFeaturesToTrack(gry1, mask=None, **ftr_params)
    mask=np.zeros_like(frame)

    while(end_flag):
        # グレースケールに変換
        gry2=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ftr1, img, mask=lucas(gry1, gry2, ftr1, mask, frame)
        gry1=gry2.copy()
        # ウィンドウに表示
        cv2.imshow('window', img)

        # ESCキー押下で終了
        if cv2.waitKey(30) & 0xff == 27:
            break

        end_flag, frame=cap.read()
    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
