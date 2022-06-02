from AIGamePlatform import Othello
from othello.bots.DeepLearning import BOT
import time


app = Othello()  # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw
bot = BOT(8)


@app.competition(competition_id='test3')
def _callback_(board, color): # 函數名稱可以自訂，board是當前盤面，color代表黑子或白子
    time.sleep(0.1)
    return bot.getAction(board, color) # 回傳要落子的座標

