import cv2
import numpy as np
import glob,os
from PIL import Image
import copy
import matplotlib.pyplot as plt


class Mascara():
    def __init__(self,imgBin,path=""):
        self.image_binario = imgBin
        self.path = path
        self.elyp_list = []#Lista de imagem-eclipse
        self.elyp_list_masc = [] #Lista de mascara-eclipse
        self.fg_list = []
        self.mask_list = []
        self.color  = (255, 255, 255)
        self.image_color= self.carregar_imagem()

    def suavizar(img):
        return cv2.GaussianBlur(img, (3, 3), 0)

    def mascara_externa(self,img):

        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)  # +cv2.THRESH_OTSU)

        # cv2.imshow('Prenchimento', thresh[::6,::6])
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print("Number of Contours found = " + str(len(contours)))
        mask = np.ascontiguousarray(img, dtype=np.uint8)
        cv2.fillPoly(mask, contours, color=(255, 255, 255))
        # cv2.imshow('Prenchimento', mask[::6,::6])
        # cv2.waitKey(0)
        return mask, contours

    def carregar_imagem(self):
        #IMAGEM DE ENTRADA
        dir = self.path
        #dir=r'C:\Users\Igor Santos\Desktop\data\projeto\imagens'
        for arquivo in glob.glob(dir+'/*.jpg'):
            print('SIM')
            dir = arquivo
            print(arquivo)

        #print(dir)

        img = cv2.imread(dir)
        # cv2.imshow("TESTE", img[::6,::6])
        # cv2.waitKey(0)
        self.image_color = img
        return img
    
    def preproc(self,x,limiar):
        #PRÉPROCESSAMENTO   -> CORREÇÃO GAMMA, SUAVIAÇÃO E LIAMIARIZAÇÃO
        #img = self.carregar_imagem()
        img = Mascara.suavizar(x)
        img = np.array(255 * (x / 255) ** 1.4, dtype='uint8')
        img_hvs = cv2.cvtColor(img,
                                 cv2.COLOR_BGR2HSV)  # COLOR_BGR2RGB  COLOR_BGR2HSV
        #img_copy = copy.copy(img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # COLOR_BGR2RGB  COLOR_BGR2HSV
        #Separa Canais
        h, s, v = cv2.split(img_hvs)

        #Limiar
        #limiar=40
        _,thresh = cv2.threshold(s,limiar,255,cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        #def detect_contorno():
        return thresh,img_gray

    def adiciona_mascara_externa(self,img_gray,thresh):
        # MASCARA EXTANA  -> DETECÇÃO DE CONTORNO, ADIÇÃO
        print(type(img_gray))
        mask,contor = self.mascara_externa(img_gray)
        #mask_inv = cv2.bitwise_not(mask)
        img_com_mascara = cv2.bitwise_and(thresh, mask)

        #LIMIAR APOS JUNÇÃO
        _,thres = cv2.threshold(img_com_mascara,155,255,cv2.THRESH_BINARY)

        mask_morf = self.morfologia(thres)
        return mask_morf
        #return mask_morf

    ######## PÓS PROCESSAMENTO ->  OPERAÇÂO MORFOLÓGICA NA MASCARA
    def morfologia(self,thres):
        mask_final =  cv2.morphologyEx(thres, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17)))
        mask_final =  cv2.morphologyEx(mask_final, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
        mask_final2 =  cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))

       # im = cv2.imread('arquivo')
        #join.show()
        #Apicação da mascara na imagem
        #join = cv2.bitwise_and(im, im, mask = mask_final)

        # cv2.imshow('Imagem Join', mask_final[::6,::6])
        # cv2.imshow("thresh", thres[::6,::6])
        # cv2.imshow("+close", mask_final2[::6,::6])
        # cv2.waitKey(0)

        return mask_final

    def corte_box(self,img, mask, contrs):
        list = []
        #ret = None
        for cnt in contrs:
            # Encontra coordenadas
            x, y, w, h = cv2.boundingRect(cnt)
            x -= 59
            y -= 59
            w += 123
            h += 120
            # Lista de imagens cortadas
          
            list.append(mask[y:y + h, x:x + w])
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
           
        return (rect, list)

    def ellipsefit(self,contours,imgbin,join):

        imcopy = self.image_color
        imcopy2 = copy.copy(imcopy)
       # join = cv2.bitwise_and(self.image_color, imcopy, mask=opening)

        for i, c in enumerate(contours):
            # contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            # boundRect[i] = cv.boundingRect(contours_poly[i])
            mask = np.zeros(imgbin.shape[:2], dtype=np.uint8)
            try:
                
                # FITELLIPSE
                ellipse = cv2.fitEllipse(c)

                ellpT = (list(ellipse))#Converte tupla para lista
                ellpT[1] = list(ellipse[1])
                # print(type(ellpT[1]))
                ellpT[1][0] += 95  # Altura
                ellpT[1][1] += 80  # Largura
                ellipse2 = tuple(ellpT)#Converte em tupla

                # Encontra coordenadas bounding box ###################
                x, y, w, h = cv2.boundingRect(c)
                x -= 70
                y -= 70
                w += 135
                h += 137

                self.mask_list.append(join[y:y + h, x:x + w])
                # cv2.imshow("Mask list",self.mask_list[0])
                # cv2.waitKey(0)
                cv2.ellipse(imcopy, ellipse2, (255, 0, 0), 10)
                cv2.ellipse(mask, ellipse2, self.color, -1)

                a = imcopy2[y:y + h, x:x + w]
                #cv2.imshow("Imcopy Cut",a)
                #cv2.waitKey(0)

                b = mask[y:y + h, x:x + w]
                self.fg_list.append(imcopy2[y:y + h, x:x + w])
                # cv2.imshow("Mask", b)
                # cv2.waitKey(0)

                # Lista de imagens cortadas0
                finalmask = cv2.bitwise_and(a, a, mask=b)
                # cv2.imshow("Join",finalmask)
                # cv2.waitKey(0)

                self.elyp_list.append(finalmask)
                # cv2.imshow("elyp_list", self.elyp_list)
                # cv2.waitKey(0)
                
                self.elyp_list_masc.append(b)
                # cv2.imshow("elyp_list_masc", self.elyp_list_masc)
                # cv2.waitKey(0)
            except Exception:#
                #print(IndexError)
                pass

        # cv2.imshow("elyp_list",self.elyp_list[0])
        # cv2.imshow("fg_list",self.fg_list[0])
        # cv2.imshow("imcopy2",imcopy2)
        # cv2.waitKey(0)

        # print("Foreground")
        # print(len(imcopy))  
        # print("List Eclipse")
        # print(len(self.elyp_list))
        # print("List Eclipse Mask")
        # print(len(self.elyp_list_masc))

        return imcopy,self.elyp_list

    def dist_transf(self,dist_fator,imgbin):

        print("Fator",dist_fator)
        imageb_copy = copy.copy(imgbin)
        # cv2.imshow("Binary",image_copy[::6,::6])
        # cv2.waitKey(0)
        # #gray = cv2.cvtColor(imgbin, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imageb_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #DISTANCE TRANFORM
        # remoção de ruído
        kernel = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        # área de fundo segura
        sure_bg = cv2.dilate(thresh,kernel,iterations=5)
        # Encontrando a área de primeiro plano segura
        dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,(int(dist_fator)/1000)*dist_transform.max(),255,0)#29=95
        #print(dist_transform)
        # Encontrando a área de primeiro plano segura
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        #CONTORNOS
        contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        print("Number of Contours found = " + str(len(contours)))

        #Lista de imgs binarias cortadas
        rect,self.mask_list = self.corte_box(imageb_copy,thresh,contours)
        #print(self.mask_list)
        
        imcolor = copy.copy(self.image_color)
        imcolorcopy= copy.copy(imcolor)
    
        # cv2.imshow("imcolor",imcolor[::6,::6])
        # cv2.imshow("imcolorcopy",imcolorcopy[::6,::6])
        # cv2.waitKey(0)

        # Apicação da mascara na imagem
        join = cv2.bitwise_and(imcolor, imcolorcopy, mask=opening)
        #cv2.imshow("Join color",join[::6,::6])s
        #cv2.waitKey(0)
        # cv2.imshow("Join color", join[::6, ::6])
        # cv2.waitKey(0)
        # plt.imshow(imcolor)
        # plt.show()

        # Lista de imgs coloridas cortadas
        #rect2, im_list = self.corte_box(imcolor, join, contours)
        # plt.imshow(rect2, cmap='gray')
        # plt.title("Bounding Box")
        # plt.show()
        # cv2.imshow("Bounding Box", rect2[::6, ::6])
        # cv2.waitKey(0)
        return self.ellipsefit(contours,imgbin,join)

        def salvar(self,mask_fn):
            dir = self.path
            
            mask_fn.show()

            # print(type(mask_fn))

            # print(dir)
            # cv2.imshow("IMG", mask_fn[::6,::6])
            # cv2.waitKey(0)
        
            #Salva as mascaras binárias
            cv2.imwrite(dir+'mb.jpg',mask_fn)

                                                              

# dirt = Mascara(r'C:\Users\Igor Santos\Desktop\data\projeto\imagens')
# img = dirt.carregar_imagem()
# cv2.imshow("TESTE", img[::6,::6])
# cv2.waitKey(0)

# imgbin,img_gray= dirt.preproc(img,50)
# cv2.imshow("TESTE", imgbin[::6,::6])
# cv2.waitKey(0)
# #dir=r'C:\Users\Igor Santos\Desktop\data\projeto\imagens'


# print(type(img_gray))
# img_masc_ext = dirt.adiciona_mascara_externa(img_gray,imgbin)
# cv2.imshow("TESTE", img_masc_ext[::6,::6])
# cv2.waitKey(0)
# # for arquivo in glob.glob(dir+'/*.jpg'):
#     print('SIM')
#     dir = arquivo
#     print(arquivo)

#print(dir)

#img = cv2.imread(dir)
#return img