from Mascara import Mascara
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog as fd
#import tkFileDialog
from tkinter import messagebox
import cv2 as cv
import numpy as np
import predition


root = tk.Tk()
root.geometry("900x560")
text_dir = tk.StringVar()
text_dir.set("")
size = 270,315

#smask = Mascara(dir)

def localizar():
    print("Buscando...")
    filename = fd.askopenfilename()
    if len(filename) > 0:
        # print(filename)
        # print(type(filename))
        if filename is None:
            messagebox.showerror(title="Nenuma arquivo encontrado", message=filename)
        else:
            messagebox.showinfo(title="Encontrada", message=filename)

            text_dir.set(str(filename))
            dir = t1.get()
            print(text_dir)
            image = cv.imread(dir)
            
            #print(type(dir))
            
            #image = Image.fromarray(image)
            img_pil = Image.open(dir) 

            image = img_pil.resize(size, Image.ANTIALIAS)
            pic = ImageTk.PhotoImage(image)
            panelA = tk.Label(root,image=pic)
            panelA.configure(image=pic)
            panelA.image = pic
            #panelA.pack(side="left", padx=10, pady=10)
            panelA.grid(row=4,column=0, sticky="W", padx=5, pady=5)

def gera_mascara(dir, img):
    print(text_dir)
    if(dir):
        #x=1
        #print("SIM",x)
        image = Image.fromarray(img)
                
        image = image.rotate(90, Image.NEAREST, expand = 1)

        image = image.resize((size), Image.ANTIALIAS)
        pic = ImageTk.PhotoImage(image)
        panelb = tk.Label(root,image=pic)
        panelb.configure(image=pic)
        panelb.image = pic
        #panelA.pack(side="left", padx=10, pady=10)
        panelb.grid(row=4, column=1,sticky="E", padx=5, pady=5)
        #gerar(img_ext)
    return img

def slide_gera_mascara(v):
    global final
    #print("slide",v)
    dir = t1.get()#Valor de Entry
  
    img = cv.imread(dir)
    mask = Mascara(img)

    imgbin,img_gray= mask.preproc(img,int(v))
    img_ext = mask.adiciona_mascara_externa(img_gray,imgbin)
    final = gera_mascara(dir, img_ext)
    print(type(final))

    #print(text_dir)
    slide_lim.grid(row=3, column=1)

    slide_tranform(slide_seg.get())
    #gera_tranform(dir, img)
 
def open():
    dir = t1.get()
    print(dir)

    if(dir):
        # x=1
        # print(x)
        img_pil = Image.open(dir)

        image = img_pil.resize((size), Image.ANTIALIAS)
        pic = ImageTk.PhotoImage(image)
        panelA = tk.Label(root,image=pic)
        panelA.configure(image=pic)
        panelA.image = pic
        #panelA.pack(side="left", padx=10, pady=10)
        panelA.grid(row=4,column=0, sticky="W", padx=5, pady=5)
   
    slide_lim.set(40)
    slide_gera_mascara(40)
    
def slide_tranform(v):
    global fitpatches, transformada
    print("Fator",v)
    dir = t1.get()#Valor Entry
  
    img = cv.imread(dir)
    mask2 = Mascara(img,dir)
    fin = np.array(final)

    transformada,fitpatches = mask2.dist_transf(v,fin)
  
    gera_tranform(dir, transformada)
    lbl_patch = tk.Label(root,text="ovos: "+str(len(fitpatches)))
    lbl_patch.grid(row=5, column=2)

def gera_tranform(dir, img):

    if(dir):
        #sx=1
        #print("SIM",x)
        im_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = Image.fromarray(im_rgb)
        
        image = image.rotate(90, Image.NEAREST, expand = 1)

        image = image.resize((size), Image.ANTIALIAS)
        pic = ImageTk.PhotoImage(image)
        panelb = tk.Label(root,image=pic)
        panelb.configure(image=pic)
        panelb.image = pic
        #panelA.pack(side="left", padx=10, pady=10)
        panelb.grid(row=4, column=2,sticky="W", padx=3, pady=3)
        #gerar(img_ext)
    #return img

def predict():
    #size = print(len(fitpatches))
    print(type(fitpatches))
    
    pred = predition.pred(fitpatches)
    print(pred)

    x = messagebox.askyesno("Resetar","Confirmar?")
    print(x)
    lbl_fertil = tk.Label(text="Ovos Ferteis:"+str(pred[1]))
    lbl_infertil = tk.Label(text="Ovos Inferteis:"+str(pred[2]))
    lbl_fertil.grid(row=6,column=0)
    lbl_infertil.grid(row=7,column=0)

    #predition.plot(pred[0],fitpatches)

def quit():
    root.destroy()


#var_save = False

#frame1 = Frame(root)
#frame2 = Frame(root)

#Label
lbl1 = tk.Label(root, text="Parte 1")   

#Input
t1=tk.Entry(root, width=40, textvariable=text_dir)
#print(text_dir)

#BUTTON
btn_load = tk.Button(root,text="Carregar Imagem",command=localizar)
btn_classif = tk.Button(root,text="Predict",command=predict)

#SLIDE
scale_var = tk.IntVar()
scale_var.set(40)
slide_lim = tk.Scale(root, from_= 10, to=150, orient="horizontal", resolution=1,
            variable=scale_var, command=slide_gera_mascara)

scale_var2 = tk.IntVar()
#scale_var2.set(40)
slide_seg = tk.Scale(root, from_= 230, to=530   , orient="horizontal", resolution=1,
            variable=scale_var2, command=slide_tranform)

#GRID
lbl1.grid(columnspan=3, pady=20)
t1.grid(row=1,padx=5)
slide_seg.grid(row=3, column=2)
slide_lim.grid(row=3, column=1)
btn_classif.grid(row=5, column=0)
btn_load.grid(column=1, row=1,padx=2)

root.mainloop()