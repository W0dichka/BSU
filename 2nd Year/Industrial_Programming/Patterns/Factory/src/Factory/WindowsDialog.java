package Factory;

import Buttons.Button;
import Buttons.Windowsbutton;

public class WindowsDialog extends Dialog {

    @Override
    public Button createButton() {
        return new Windowsbutton();
    }
}
