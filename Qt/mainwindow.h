#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 声明浏览模型和浏览文件夹的槽函数
    void on_btn_browse_model_clicked();
    void on_btn_browse_folder_clicked();
    void on_btn_start_diagnosis_clicked();
    void readPythonOutput();

    void on_btn_continuous_monitor_clicked();

private:
    Ui::MainWindow *ui;
    QProcess *diagnosisProcess;
};

#endif // MAINWINDOW_H
