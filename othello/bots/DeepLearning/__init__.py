import numpy as np
from othello.OthelloUtil import getValidMoves
from othello.bots.DeepLearning.OthelloModel import OthelloModel
from othello.OthelloGame import OthelloGame


class BOT:

    def __init__(self, board_size, *args, **kargs):
        self.board_size=board_size
        self.model = OthelloModel(input_shape=(self.board_size, self.board_size) )
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
    
    def getAction(self, game, color):
        predict = self.model.predict( game )
        valid_positions=getValidMoves(game, color)
        valids=np.zeros((game.size), dtype='int')
        valids[ [i[0]*game.n+i[1] for i in valid_positions] ]=1
        predict*=valids
        position = np.argmax(predict)
        
        if self.collect_gaming_data:
            tmp=np.zeros_like(predict)
            tmp[position]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        
        position=(position//game.n, position%game.n)
        return position
    
    def self_play_train(self, args):
        self.collect_gaming_data=True
        def gen_data():
            def getSymmetries(board, pi):
                # mirror, rotational
                pi_board = np.reshape(pi, (len(board), len(board)))
                l = []
                for i in range(1, 5):
                    for j in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if j:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        l += [( newB, list(newPi.ravel()) )]
                return l
            self.history=[]
            history=[]
            game=OthelloGame(self.board_size)
            game.play(self, self, verbose=args['verbose'])
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b,p in sym:
                    history.append([b, p, player])
            self.history.clear()
            game_result=game.isEndGame()
            return [(x[0],x[1]) for x in history if (game_result==0 or x[2]==game_result)]
        
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            data+=gen_data()
        
        self.collect_gaming_data=False
        
        self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()
        
