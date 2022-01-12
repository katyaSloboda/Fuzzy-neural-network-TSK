# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:51:41 2020

@author: k.sloboda
"""


from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import messagebox
from tkinter import filedialog

import os
import numpy as np
import sys

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fnn_tsk import FNN_TSK

import threading
import nest_asyncio
nest_asyncio.apply()


class TSK_GUI:
    def __init__(self, master):
        self.file_path = ''
        self.fnn_tsk = ''
        self.master = master
        
        master.title("Program for building FNN TSK")
        master.geometry("600x420+400+150")

        self.info_btn = Button(master, text="Info", command=self.get_info_window)
        self.row1_lbl = Label(master, text="")
        self.header_lbl = Label(master, text="FNN TSK parameters:")
         
        # file name
        self.file_lbl = Label(master, text="Data file")
        self.file_btn = Button(master, text="Choose", command=self.open_file_dialog)
        
        # percentage of test data
        var = IntVar()
        var.set(30)
        self.test_data_percent_lbl = Label(master, text="Percentage of test data")
        self.test_data_percent = Spinbox(master, from_=10, to=90, width=18, textvariable=var)
        
        # number of rules
        var = IntVar()
        var.set(5)
        self.rules_lbl = Label(master, text="Number of rules")
        self.rules = Spinbox(master, from_=1, to=sys.maxsize, width=18, textvariable=var)
        
        # number of epochs
        var = IntVar()
        var.set(10)
        self.epochs_lbl = Label(master, text="Number of epochs")
        self.epochs = Spinbox(master, from_=1, to=sys.maxsize, width=18, textvariable=var)
        
        # number of cycles
        var = IntVar()
        var.set(3)
        self.cycles_lbl = Label(master, text="Number of cycles")
        self.cycles = Spinbox(master, from_=1, to=sys.maxsize, width=18, textvariable=var)
        
        # value of the gradient parameter "eta"
        self.eta_lbl = Label(master, text="Gradient parameter \"eta\"")
        self.eta = Entry(master, width=20)
        self.eta.insert(0,'0.001')
        
        # error threshold
        self.error_threshold_lbl = Label(master, text="Error threshold")
        self.error_threshold = Entry(master, width=20)
        self.error_threshold.insert(0,'0.00000001')
        
        self.row10_lbl = Label(master, text="")
        
        # start training
        self.start_btn = Button(master, text="Start training", command=self.start_thread_for_training)
        
        # progress bar
        self.progressbar = Progressbar(master, orient = HORIZONTAL, 
                                       length = 125, mode = 'determinate')
        
        # error
        self.error_value = StringVar()
        self.error_lbl = Label(master, text="Error: ")
        self.error_value_lbl = Label(master, textvariable=self.error_value)
        
        self.row14_lbl = Label(master, text="")
        
        # plot result
        self.plot_btn = Button(master, text="Plot results", command=self.plot_model)
        
        # plot result
        #self.plot_model_as_line_btn = Button(master, text="Plot results", command=self.plot_model_as_line)
        
        # plot errors
        self.plot_errors_btn = Button(master, text="Plot errors", command=self.plot_errors)
        
        #---------------------------------------------------------------------
        # LAYOUT
        
        self.info_btn.grid(row=0, column=0, sticky="W", padx=20)
        
        self.row1_lbl.grid(row=1, column=0, sticky="W")
        self.header_lbl.grid(row=2, column=0, sticky="W", padx=20)
        self.file_lbl.grid(row=3, column=0, sticky="W", padx=20)
        self.file_btn.grid(row=3, column=1, sticky="W")
        self.test_data_percent_lbl.grid(row=4, column=0, sticky="W", padx=20)
        self.test_data_percent.grid(row=4, column=1, sticky="W")
        self.rules_lbl.grid(row=5, column=0, sticky="W", padx=20)
        self.rules.grid(row=5, column=1, sticky="W")
        self.epochs_lbl.grid(row=6, column=0, sticky="W", padx=20)
        self.epochs.grid(row=6, column=1, sticky="W")
        self.cycles_lbl.grid(row=7, column=0, sticky="W", padx=20)
        self.cycles.grid(row=7, column=1, sticky="W")
        self.eta_lbl.grid(row=8, column=0, sticky="W", padx=20)
        self.eta.grid(row=8, column=1, sticky="W")
        self.error_threshold_lbl.grid(row=9, column=0, sticky="W", padx=20)
        self.error_threshold.grid(row=9, column=1, sticky="W")
        
        self.row10_lbl.grid(row=10, column=0, sticky="W", padx=20)
        self.start_btn.grid(row=11, column=0, sticky="W", padx=20)
        self.progressbar.grid(row=12, column=0, sticky="W", padx=20, pady=5)
        self.error_lbl.grid(row=13, column=0, sticky="W", padx=20)
        self.error_value_lbl.grid(row=13, column=1, sticky="W")
        
        self.row14_lbl.grid(row=14, column=0, sticky="W", padx=20)
        self.plot_btn.grid(row=15, column=0, sticky="W", padx=20, pady=5)
        #self.plot_model_as_line_btn.grid(row=16, column=0, sticky="W", padx=20, pady=5)
        self.plot_errors_btn.grid(row=16, column=0, sticky="W", padx=20)
        
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            try:
                self.thread.exit()
            except:
                pass
            finally:
                self.master.destroy()
        
            
    def get_info_window(self):
        messagebox.showinfo("Info", ('This application is designed to build and train the FNN TSK model.\n\n' + 
                                     'There are such parameters:\n' +
                                     '1. Rules - the number of nodes in the neural network\n' +
                                     '2. Epochs - the number of training cycles on training data\n' +
                                     '3. Cycles - the number of repetitions during training for the MF ' + 
                                     'parameters (i.e., the number of repetitions of the second stage with' +
                                     'one execution of the first)\n' +
                                     '4. \"Eta\" is a parameter for gradient descent\n' +
                                     '5. Error threshold - the threshold at which the model stops learning\n\n' +
                                     'Error - error value on test data\n\n' +
                                     'The graph shows the real and predicted output values ​​for each class' +
                                     '(1, -1) for training and test data'))
        
        
    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes = (("CSV files","*.csv"),("all files","*.*")),
                                               initialdir= os.path.dirname(__file__))
        if file_path:
            self.file_path = file_path
            Label(self.master, text=self.file_path).grid(row=3, column=2, sticky="W")
        
        
    def disable_buttons(self):
        self.start_btn['state'] = 'disabled'
        self.plot_btn['state'] = 'disabled'
        self.plot_errors_btn['state'] = 'disabled'
        self.master.update_idletasks()
        
        
    def enable_buttons(self):
        self.start_btn['state'] = 'normal'
        self.plot_btn['state'] = 'normal'
        self.plot_errors_btn['state'] = 'normal'
        self.master.update_idletasks()
    
    
    def start_thread_for_training(self):
        self.thread = threading.Thread(target=self.start_train)
        self.thread.start()
        
        
    def start_train(self):
        self.error_value.set('')
        self.update_progressbar(0)
        
        if not self.file_path:
            messagebox.showwarning("Warning", "Choose data file")
            return
        
        test_data_percent = int(self.test_data_percent.get())
        rules = int(self.rules.get())
        epochs = int(self.epochs.get())
        cycles = int(self.cycles.get())
        eta = float(self.eta.get())
        error_threshold = float(self.error_threshold.get())
        
        print("\n------------------------",
              "\n---Start new training---",
              "\ntest_data_percent: ", test_data_percent,
              "\nrules: ", rules,
              "\nepochs: ", epochs,
              "\ncycles: ", cycles,
              "\neta: ", eta,
              "\nerror_threshold: ", error_threshold)
        
        if (test_data_percent < 10 or test_data_percent > 90 or
            rules < 1 or epochs < 1 or cycles < 1 or
            eta <= 0 or eta >= 1 or
            error_threshold <= 0 or error_threshold >= 1):
            messagebox.showwarning("Warning", "Invalid parameter values")
            return
        
        self.fnn_tsk = FNN_TSK(self.file_path, 
                               test_data_percent, rules, epochs, 
                               cycles, eta, error_threshold)
        
        self.disable_buttons()
        
        self.train_model_and_update_progressbar()
        self.fnn_tsk.test_model()
        self.error_value.set(self.fnn_tsk.test_error)
        
        self.enable_buttons()
        self.print_errors()
        
        messagebox.showinfo("Info", "Training completed")
        
        
        
    def train_model_and_update_progressbar(self):
        if not self.fnn_tsk:
            messagebox.showwarning("Warning", "Train FNN TSK model")
            return
        
        for i in self.fnn_tsk.train_model():
            self.update_progressbar(i)
        self.update_progressbar(100)
        
        
    def update_progressbar(self, value):
        self.progressbar['value'] = value
        self.master.update_idletasks()
        
        
    def print_errors(self):
        firstErrors = 0
        secondErrors = 0
                
        firstErrors += sum([1 for i in range(self.fnn_tsk.D_train.__len__()) if self.fnn_tsk.D_train[i] < 0 and self.fnn_tsk.Y_train[i] >= 0])
        firstErrors += sum([1 for i in range(self.fnn_tsk.D_test.__len__()) if self.fnn_tsk.D_test[i] < 0 and self.fnn_tsk.Y_test[i] >= 0])
        secondErrors += sum([1 for i in range(self.fnn_tsk.D_train.__len__()) if self.fnn_tsk.D_train[i] > 0 and self.fnn_tsk.Y_train[i] <= 0])
        secondErrors += sum([1 for i in range(self.fnn_tsk.D_test.__len__()) if self.fnn_tsk.D_test[i] > 0 and self.fnn_tsk.Y_test[i] <= 0])
        
        print("First errors: ", firstErrors)
        print("Second errors: ", secondErrors)
        

    def plot_model(self):
        if (not self.fnn_tsk or
            not self.fnn_tsk.Y_train or
            not self.fnn_tsk.D_train or
            not self.fnn_tsk.Y_test or
            not self.fnn_tsk.D_test):
            messagebox.showwarning("Warning", "Train FNN TSK model")
            return
                  
        plot_window = Tk()
        plot_window.title("Plot of FNN TSK model")
                        
        train_fig = self.get_figure(np.array([t for t in range(len(self.fnn_tsk.D_train))]), 
                                    np.array(self.fnn_tsk.D_train), 
                                    np.array([t for t in range(len(self.fnn_tsk.Y_train))]), 
                                    np.array([(t if t>-2 else -2) if t<2 else 2
                                             for t in self.fnn_tsk.Y_train]), 
                                    "Train data")
                                
        train_canvas = FigureCanvasTkAgg(train_fig, master=plot_window)
        train_canvas.get_tk_widget().grid(row=0, column=0, sticky="W")
        train_canvas.draw()
                
        test_fig = self.get_figure(np.array([t for t in range(len(self.fnn_tsk.D_test))]), 
                                   np.array(self.fnn_tsk.D_test),
                                   np.array([t for t in range(len(self.fnn_tsk.Y_test))]), 
                                   np.array([(t if t>-2 else -2) if t<2 else 2
                                             for t in self.fnn_tsk.Y_test]),
                                   "Test data")
                                
        test_canvas = FigureCanvasTkAgg(test_fig, master=plot_window)
        test_canvas.get_tk_widget().grid(row=0, column=1, sticky="W")
        test_canvas.draw()
                                
        plot_window.mainloop()
        
        
    def get_figure(self, x_real, y_real, x_predict, y_predict, title):
        class1_real = [i for i, v in enumerate(y_real) if v == 1]
        
        class1_x_real = [v for i, v in enumerate(x_real) if i in class1_real]
        class1_y_real = [v for i, v in enumerate(y_real) if i in class1_real]
        class2_x_real = [v for i, v in enumerate(x_real) if i not in class1_real]
        class2_y_real = [v for i, v in enumerate(y_real) if i not in class1_real]
        
        class1_x_predict = [v for i, v in enumerate(x_predict) if i in class1_real]
        class1_y_predict = [v for i, v in enumerate(y_predict) if i in class1_real]
        class2_x_predict = [v for i, v in enumerate(x_predict) if i not in class1_real]
        class2_y_predict = [v for i, v in enumerate(y_predict) if i not in class1_real]
        
        fig = Figure(figsize=(6,6))
        subplot = fig.add_subplot(111)
        subplot.plot(class1_x_real,    class1_y_real,    'bo', label='real c1', alpha=0.5)
        subplot.plot(class1_x_predict, class1_y_predict, 'r*', label='predict c1', alpha=0.5)
        subplot.plot(class2_x_real,    class2_y_real,    'go', label='real c2', alpha=0.5)
        subplot.plot(class2_x_predict, class2_y_predict, 'k*', label='predict c2', alpha=0.5)
        subplot.plot(x_real, [0 for t in range(len(y_real))], 'k-')
                                
        subplot.set_title (title, fontsize=14)
        subplot.set_ylabel("Y", fontsize=12)
        subplot.set_xlabel("X", fontsize=12)
        subplot.legend(fontsize=10)
        subplot.grid()
        return fig
        

    def plot_errors(self):
        if (not self.fnn_tsk or
            not self.fnn_tsk.Y_train or
            not self.fnn_tsk.D_train or
            not self.fnn_tsk.Y_test or
            not self.fnn_tsk.D_test):
            messagebox.showwarning("Warning", "Train FNN TSK model")
            return
                  
        errors_window = Tk()
        errors_window.title("Plot of FNN TSK model errors")
             
        x = [i for i in range(self.fnn_tsk.train_errors.__len__())]
        print(self.fnn_tsk.train_errors)
        
        fig = Figure(figsize=(6,6))
        subplot = fig.add_subplot(111)
        subplot.plot(x, self.fnn_tsk.train_errors, 'k-')
                                
        subplot.set_ylabel("Error", fontsize=12)
        subplot.set_xlabel("Epoch", fontsize=12)
        subplot.grid()
                                
        train_canvas = FigureCanvasTkAgg(fig, master=errors_window)
        train_canvas.get_tk_widget().grid(row=0, column=0, sticky="W")
        train_canvas.draw()
                                
        errors_window.mainloop()
    


def main():
    root = Tk()
    my_gui = TSK_GUI(root)
    root.protocol("WM_DELETE_WINDOW", my_gui.on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
    
    
    