import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        FileWriter fr = new FileWriter("output.txt");
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        String text = new String();
        String text2 = new String();
        String kov = new String();

        while (scan.hasNextLine()) {
            text2 = scan.nextLine();
            text += text2;
            text += '\n';
        }

        enum State {Start, Str_Const, Str_Const2, Com_Beg, Com_Read, Com_Mul, Com_End, Symb_Const, Ctrl_Symb, Symb_End, Ctrl_SymbC, Ctrl_SymbS}
        State state = State.Start;
        List<String> result = new ArrayList<>();
        StringBuilder fieldValue = new StringBuilder();
        for (char c : text.toCharArray()) {
            switch (state) {
                case Start:
                    if (c == '"') {
                        fieldValue.append(c);
                        state = State.Str_Const;
                    } else if (c == '/') {
                        state = State.Com_Beg;
                    } else if (c == '\'') {
                        fieldValue.append(c);
                        state = State.Symb_Const;
                    } else {
                        fieldValue.append(c);
                        state = State.Start;
                    }
                    break;
                case Str_Const:
                    if (c == '"') {
                        fieldValue.append(c);
                        state = State.Str_Const2;
                    } else if (c == '\\') {
                        fieldValue.append(c);
                        state = state.Ctrl_SymbS;
                    } else {
                        fieldValue.append(c);
                        state = State.Str_Const;
                    }
                    break;
                case Str_Const2:
                    if (c == '"') {
                        fieldValue.append(c);
                        state = State.Str_Const;
                    }
                    else if(c == '/'){
                        state = State.Com_Beg;
                    }
                    else {
                        fieldValue.append(c);
                        state = State.Start;
                    }
                    break;
                case Ctrl_SymbS:
                    fieldValue.append(c);
                    state = State.Str_Const;
                    break;
                case Symb_Const:
                    if (c == '\\') {
                        fieldValue.append(c);
                        state = State.Ctrl_SymbC;
                    } else {
                        fieldValue.append(c);
                        state = State.Symb_End;
                    }
                    break;
                case Ctrl_SymbC:
                    fieldValue.append(c);
                    state = State.Symb_Const;
                    break;
                case Symb_End:
                    fieldValue.append(c);
                    state = State.Start;
                    break;
                case Com_Beg:
                    if (c == '/')
                        state = State.Com_Read;
                    else if (c == '*')
                        state = State.Com_Mul;
                    else {
                        fieldValue.append(c);
                        state = State.Start;
                    }
                    break;
                case Com_Read:
                    if (c == '\n') {
                        fieldValue.append(c);
                        state = State.Start;
                    } else {
                        state = State.Com_Read;
                    }
                    break;
                case Com_Mul:
                    if (c == '*')
                        state = State.Com_End;
                    else {
                        state = State.Com_Mul;
                    }
                    break;
                case Com_End:
                    if (c == '/')
                        state = State.Start;
                    else {
                        state = State.Com_Mul;
                    }
                    break;
            }
        }
            result.add(fieldValue.toString());
            for (int i = 0; i < result.size(); i++) {
                String temp = result.get(i);
                fr.write(temp);
            }
            gr.close();
            fr.close();
        }
}