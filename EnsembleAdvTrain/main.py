from adversarial_vision_challenge import model_server
import vgg19

if __name__ == '__main__':
    
    model = vgg19.Vgg19([122.46267559570313	114.25840612792969	101.3746757055664],'./models/vgg19-pre.npy')
    
    model_server(model)
