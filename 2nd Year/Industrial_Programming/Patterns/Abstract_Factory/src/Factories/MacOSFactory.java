package Factories;

import Buttons.Button;
import Buttons.MacOSButton;
import Checkboxes.Checkbox;
import Checkboxes.MacOSCheckbox;

public class MacOSFactory implements GUIFactory {
    @Override
    public Button createButton(){
        return new MacOSButton();
    }
    @Override
    public Checkbox createCheckbox(){
        return new MacOSCheckbox();
    }
}
