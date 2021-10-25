import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image

from src.main import main
from validate_state import Validator
import os

run_directory = os.getcwd()


def menu():
    global root
    global variant
    global path_name
    global folder_path
    global file_name
    global checkbutton_png
    global checkbutton_pgf

    root = tk.Tk()

    window_setup()
    image_place()
    variant_box = BoxWidget("Вариант:")
    variant = variant_box.take_text()
    folder_path = tk.StringVar()
    browser_box()
    checkbutton_png = CheckbuttonWidget("png", "Сохранить графики в формате:")
    checkbutton_pgf = CheckbuttonWidget("pgf", exe_function=pgf_warning)
    button_setup()
    checkbutton_png.button.grid()
    checkbutton_pgf.button.grid()

    root.mainloop()


def window_setup():
    root.title("Coursework")
    window_width = 400
    window_height = 600
    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)


def button_setup():
    button_icon = Image.open("../assets/button_img.png")
    button_icon = button_icon.resize((40, 40), Image.ANTIALIAS)
    button_icon = ImageTk.PhotoImage(button_icon)

    demo_button = ttk.Button(
        root,
        image=button_icon,
        text="Запустить",
        compound=tk.LEFT,
        command=press_button,
    )
    demo_button.button_icon = button_icon
    demo_button.grid()
    return demo_button


def image_place():
    image = Image.open("../assets/top_logo.png")
    image = image.resize((125, 125), Image.ANTIALIAS)
    my_img = ImageTk.PhotoImage(image)
    place = tk.Label(image=my_img)
    place.my_img = my_img
    place.grid()


class CheckbuttonWidget:
    def __init__(self, text, text_above=None, exe_function=None):
        self.space = ttk.Frame(root)
        self.space.grid(ipady=2, sticky="n")
        if text_above:
            above_box_text = ttk.Label(self.space, text=str(text_above))
            above_box_text.grid()
        self.checkmark_state = tk.IntVar()
        self.button = ttk.Checkbutton(
            self.space,
            text=str(text),
            variable=self.checkmark_state,
            command=exe_function,
            onvalue=1,
            offvalue=0,
            width=5,
        )


class BoxWidget:
    def __init__(self, text):
        self.space = ttk.Frame(root)
        self.above_box_text = ttk.Label(self.space, text=str(text))

    def take_text(self):
        value = tk.StringVar()
        vcmd = (self.space.register(self.only_num_vaild), "%S")
        box_entry = ttk.Entry(
            self.space, validate="key", validatecommand=vcmd, textvariable=value
        )
        self.space.grid(sticky="w", padx=39)
        self.above_box_text.grid(sticky="w")
        box_entry.grid(sticky="w")
        return value

    def only_num_vaild(self, S):
        if S in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return True
        self.space.bell()  # .bell() plays that ding sound telling you there was invalid input
        return False


def browser_box():
    space = ttk.Frame(root)
    above_text = ttk.Label(space, text="Сохранить как:")
    above_text.grid(sticky="w")

    info = above_text.grid_info()
    row = info["row"]
    column = info["column"]

    box_entry = ttk.Entry(space, textvariable=folder_path, width=40)
    folder_button = ttk.Button(space, text="Browse", command=browse_button)

    space.grid()
    box_entry.grid(row=row + 1, column=column)
    folder_button.grid(row=row + 1, column=column + 1)


def browse_button():
    file_name = filedialog.askdirectory()
    folder_path.set(file_name)


def press_button():
    save_names = []
    os.chdir(run_directory)
    pgf_state = checkbutton_pgf.checkmark_state.get()
    png_state = checkbutton_png.checkmark_state.get()
    variant_number = variant
    path = folder_path.get()

    check = Validator(variant_number, path, pgf_state, png_state)
    if not check.variant_validate():
        variant_error()

    if not check.path_validate():
        path_error()

    if check.pgf_state_validate():
        save_names.append("pgf")

    if check.png_state_validate():
        save_names.append("png")

    if check.variant_validate() and check.path_validate():
        int_variant = int(variant_number.get())
        if not save_names:
            user_answer = dont_save_plot_warning()
            if user_answer == "yes":
                main(int_variant, save_names, path)
            else:
                pass
            return
        main(int_variant, save_names, path)


def variant_error():
    messagebox.showerror("Error", "Неверный вариант!\n(Доступные варианты с 1 - 24)")


def path_error():
    messagebox.showerror("Error", "Неверный путь сохранения!")


def dont_save_plot_warning():
    state = messagebox.askquestion(
        "Warning",
        "Вы не выбрали сохранение графиков.\nЕсли продолжить графики не будут сохраняться.",
        icon="warning",
    )
    return state


def pgf_warning():
    pgf_status = checkbutton_pgf.checkmark_state.get()
    if pgf_status:
        user_answer = run_time_warning()
        if user_answer == "no":
            checkbutton_pgf.checkmark_state.set(0)


def run_time_warning():
    state = messagebox.askquestion(
        "Warning",
        "Файлы в pgf для LaTeX сохраняются очень долго :( \nХочешь сохранить?",
        icon="warning",
    )
    return state


if __name__ == "__main__":
    menu()
