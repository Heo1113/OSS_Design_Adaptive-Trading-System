package ats;

import ats.presentation.MainFrame;

import javax.swing.SwingUtilities;

/**
 * 진입점.
 *   java -jar ATS.jar          → GUI 실행
 *   java -jar ATS.jar --cli    → 콘솔 데모 (헤드리스 환경 검증용)
 */
public class Main {
    public static void main(String[] args) {
        for (String a : args) {
            if (a.equals("--cli")) {
                ConsoleDemo.run();
                return;
            }
        }
        SwingUtilities.invokeLater(() -> new MainFrame().setVisible(true));
    }
}
