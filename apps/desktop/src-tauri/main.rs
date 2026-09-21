use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            // Overlay compacto, sempre no topo, canto direito.
            if let Some(win) = app.get_webview_window("overlay") {
                let _ = win.set_always_on_top(true);
                #[cfg(target_os = "macos")]
                let _ = win.set_position(tauri::Position::Physical(tauri::PhysicalPosition {
                    x: 20,
                    y: 40,
                }));
            }
            // Hotkey global: Ctrl+Shift+C copia a melhor frase do último card EN.
            let handle = app.handle().clone();
            let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyC);
            let _ = handle.global_shortcut().register(shortcut);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("erro ao iniciar Meeting Copilot");
}
