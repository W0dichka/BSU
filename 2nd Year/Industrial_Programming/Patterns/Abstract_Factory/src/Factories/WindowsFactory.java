package Factories;

import Buttons.Button;
import Buttons.WindowsButton;
import Checkboxes.Checkbox;
import Checkboxes.WindowsCheckbox;

public class WindowsFactory implements  GUIFactory{
    @Override
    public Button createButton(){
        return new WindowsButton();
    }
    @Override
    public Checkbox createCheckbox(){
        return new WindowsCheckbox();
    }
}
