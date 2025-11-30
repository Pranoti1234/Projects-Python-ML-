from  tkinter import *
def click(event): #for button bind event argument is compalsary
    global scvalue
    #event.widget gives a button which we click 
    #.cget function which gives text of button which we click
    text = event.widget.cget("text")
    print(text)
    if text == "=":
        if scvalue.get().isdigit():
            value = int(scvalue.get())
        else:
            try:
                value = eval(screen.get()) #eval function evaluate string like 2x5=10
            except Exception as e:
                print(e)
                value="Error"
                
        

        scvalue.set(value)
        screen.update()

    elif text == "C":
        scvalue.set("")
        screen.update()
    else:
        scvalue.set(scvalue.get() + text)
        screen.update()#here update the value of scvalue with new value
        
root = Tk()
root.geometry("644x900")
root.title("Calculator")

scvalue = StringVar()
scvalue.set("")

frame = Frame(root, bg="black")
frame.pack()

screen = Entry(frame, textvar=scvalue, font="lucida 35 bold")#Entry function allows a single line text
screen.pack(ipadx=8, pady=10, padx=10)#ipadx=internalpadding



f = Frame(frame, bg="grey")
for i in range(9,6,-1):
    b = Button(f, text=str(i), padx=24, pady=18, font="lucida 25 bold")
    b.pack(side=LEFT, padx=19, pady=8)
    b.bind("<Button-1>", click)
b = Button(f, text="+", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=19, pady=8)
b.bind("<Button-1>", click)

f.pack()

f = Frame(frame, bg="grey")
for i in range(6,3,-1):
    b = Button(f, text=str(i), padx=24, pady=18, font="lucida 25 bold")
    b.pack(side=LEFT, padx=20, pady=8)
    b.bind("<Button-1>", click)
b = Button(f, text="-", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=20, pady=8)
b.bind("<Button-1>", click)
f.pack()


f = Frame(frame, bg="grey")
for i in range(3,0,-1):
    b = Button(f, text=str(i), padx=24, pady=18, font="lucida 25 bold")
    b.pack(side=LEFT, padx=19.8, pady=8)
    b.bind("<Button-1>", click)
b = Button(f, text="*", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=19.8, pady=8)
b.bind("<Button-1>", click)
f.pack()


f = Frame(frame, bg="grey")
b = Button(f, text="C", padx=26, pady=18, font="lucida 25 bold",bg="red",fg="white")
b.pack(side=LEFT, padx=17.3, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="0", padx=26, pady=18, font="lucida 25 bold")
b.pack(side=LEFT, padx=17.3, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="=", padx=26, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=17.3, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="/", padx=26, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=17.3, pady=8)
b.bind("<Button-1>", click)
f.pack()


f = Frame(frame, bg="grey")
b = Button(f, text=".", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=20, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="%", padx=24, pady=18, font="lucida 22 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=20, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="#", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=20, pady=8)
b.bind("<Button-1>", click)
b = Button(f, text="!", padx=24, pady=18, font="lucida 25 bold",bg="black",fg="white")
b.pack(side=LEFT, padx=20, pady=8)
b.bind("<Button-1>", click)
f.pack()

root.mainloop()
