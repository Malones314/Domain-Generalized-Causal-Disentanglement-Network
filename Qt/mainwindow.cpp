#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug>
#include <QSettings>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	this->setWindowTitle(tr("智能声学故障诊断系统"));
    ui->tableWidget_results->setColumnCount(4);
    ui->tableWidget_results->setHorizontalHeaderLabels(QStringList() << "文件名" << "分析状态" << "诊断结果" << "置信度");
    ui->tableWidget_results->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->tableWidget_results->setEditTriggers(QAbstractItemView::NoEditTriggers); // 设置为只读模式

    diagnosisProcess = new QProcess(this);
    connect(diagnosisProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::readPythonOutput);
    // 拦截 Python 运行时的崩溃或红字报错
    connect(diagnosisProcess, &QProcess::readyReadStandardError, this, [this]() {
        QByteArray errorData = diagnosisProcess->readAllStandardError();
        // 如果控制台乱码，可以尝试 QString::fromLocal8Bit(errorData)
        qDebug() << "【Python致命报错】:" << QString::fromUtf8(errorData);
    });
}

MainWindow::~MainWindow()
{
    if(diagnosisProcess->state() == QProcess::Running) {
        diagnosisProcess->kill();
        diagnosisProcess->waitForFinished();
    }
    delete ui;
}

void MainWindow::on_btn_browse_model_clicked()
{
    // 唤起文件选择框，指定过滤条件为 .pth 模型文件

    QString modelPath = QFileDialog::getOpenFileName(this,
                                                     tr("选择模型文件"),
                                                     QDir::currentPath(),
                                                     tr("模型文件 (*.pth);;所有文件 (*.*)"));

    // 如果用户没有取消选择，则将路径显示在对应的文本框中
    if (!modelPath.isEmpty()) {
        ui->lineEdit_model->setText(modelPath);
    }
}

void MainWindow::on_btn_browse_folder_clicked()
{
    // 唤起文件夹选择框，设置选项为仅显示目录

    QString folderPath = QFileDialog::getExistingDirectory(this,
                                                           tr("选择待测试文件夹"),
                                                           QDir::currentPath(),
                                                           QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);


    // 如果用户选中了文件夹，则将路径显示在对应的文本框中
    if (!folderPath.isEmpty()) {
        ui->lineEdit_folder->setText(folderPath);
    }
}

void MainWindow::on_btn_start_diagnosis_clicked()
{
    ui->progressBar->setValue(0);
    ui->label_status->setText(tr("初始化模型环境中..."));
    QString modelPath = ui->lineEdit_model->text();
    QString folderPath = ui->lineEdit_folder->text();

    if (modelPath.isEmpty() || folderPath.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), tr("请先选择模型文件和待测试文件夹！"));
        return;
    }

    if (diagnosisProcess->state() == QProcess::Running) {
        QMessageBox::information(this, tr("提示"), tr("诊断正在进行中，请耐心等待..."));
        return;
    }

    QString configPath = QDir(QCoreApplication::applicationDirPath()).filePath("config.ini");
    qDebug()<< configPath;
    QSettings settings(configPath, QSettings::IniFormat);

    // 读取配置，如果没填则给个空字符串或默认值
    QString pythonExe = settings.value("Environment/PythonExe", "").toString();
    QString algorithmDir = settings.value("Environment/AlgorithmDir", "").toString();

    // 鲁棒性检查：如果配置文件里的路径为空，拦截并提示
    if (pythonExe.isEmpty() || algorithmDir.isEmpty()) {
        QMessageBox::critical(this, tr("配置错误"), tr("请在 config.ini 中正确配置 PythonExe 和 AlgorithmDir！"));
        return;
    }

    ui->btn_start_diagnosis->setEnabled(false);
    ui->btn_start_diagnosis->setText(tr("正在诊断..."));

    // 强制使用 config.ini 中读取到的 AlgorithmDir 作为工作目录和脚本路径
    QDir algoDir(algorithmDir);
    QString scriptPath = algoDir.absoluteFilePath("run_diagnosis.py");

    // 【核心修复】：设置 Python 进程的工作目录，确保它能找到同级目录下的 CBGN.py 等依赖
    diagnosisProcess->setWorkingDirectory(algoDir.absolutePath());

    QStringList arguments;
    // 在参数最前面加上 "-u"，强制 Python 禁用输出缓冲
    arguments << "-u"
              << scriptPath
              << "--model_path"
              << modelPath
              << "--folder_path"
              << folderPath;

    // 打印调试信息，确认路径是否正确
    qDebug() << "准备执行程序:" << pythonExe;
    qDebug() << "参数列表:" << arguments;
    qDebug() << "工作目录:" << diagnosisProcess->workingDirectory();

    // 新增：强制 Python 输出 UTF-8，解决控制台中文乱码
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONIOENCODING", "utf-8");
    diagnosisProcess->setProcessEnvironment(env);

    diagnosisProcess->start(pythonExe, arguments);
}

void MainWindow::readPythonOutput()
{
    // 只要有完整的一行数据到达，就循环读取
    while (diagnosisProcess->canReadLine()) {
        // 读取一行并去掉末尾的回车换行符
        QString line = QString::fromUtf8(diagnosisProcess->readLine()).trimmed();

        // 1. 判断是否是我们约定的进度标记
        if (line.startsWith("PROGRESS:")) {
            // 切割字符串，格式为 PROGRESS:当前:总数:文件名
            QStringList parts = line.split(":");
            if (parts.size() >= 4) {
                int current = parts[1].toInt();
                int total = parts[2].toInt();
                // 如果文件名本身包含冒号，把后面的部分重新拼起来
                QString filename = parts.mid(3).join(":");

                // 更新UI进度条和状态文本
                ui->progressBar->setMaximum(total);
                ui->progressBar->setValue(current);
                ui->label_status->setText(QString("正在分析: %1 (%2/%3)").arg(filename).arg(current).arg(total));
            }
        }
        // 【新增拦截】：处理 Python 发来的实时调试信息
        else if (line.startsWith("DEBUG:")) {
            // 截取 "DEBUG: " 后面的实际内容
            QString debugMsg = line.mid(6).trimmed();
            // 实时更新到界面的状态标签上
            ui->label_status->setText(debugMsg);
            // 同时打印到控制台，方便在 Qt Creator 里排查
            qDebug() << "后端状态:" << debugMsg;
        }
        // 2. 判断是否是最终输出的 JSON 结果 (通常以 '{' 开头)
        else if (line.startsWith("{")) {
            QJsonParseError parseError;
            QJsonDocument jsonDoc = QJsonDocument::fromJson(line.toUtf8(), &parseError);

            if (parseError.error == QJsonParseError::NoError && jsonDoc.isObject()) {
                QJsonObject rootObj = jsonDoc.object();
                if (rootObj.contains("error")) {
                    QMessageBox::critical(this, tr("诊断失败"), rootObj["error"].toString());
                    ui->label_status->setText(tr("诊断中止"));
                } else if (rootObj.contains("results")) {
                    QJsonArray resultsArray = rootObj["results"].toArray();
                    ui->tableWidget_results->setRowCount(0); // 每次填充前清空旧数据

                    for(int i = 0; i < resultsArray.size(); ++i) {
                        QJsonObject item = resultsArray[i].toObject();
                        ui->tableWidget_results->insertRow(i); // 在表格末尾插入新行

                        QString filename = item["filename"].toString();
                        QString status = item["status"].toString();
                        // 如果成功则提取预测结果，失败则提取报错信息
                        QString prediction = item.contains("prediction") ? item["prediction"].toString() : item["message"].toString();
                        // 提取置信度，如果没有则显示横杠
                        QString confidence = item.contains("confidence") ? QString::number(item["confidence"].toDouble(), 'f', 4) : "-";

                        QTableWidgetItem *fileItem = new QTableWidgetItem(filename);
                        QTableWidgetItem *statusItem = new QTableWidgetItem(status == "success" ? "成功" : "失败");
                        QTableWidgetItem *predItem = new QTableWidgetItem(prediction);
                        QTableWidgetItem *confItem = new QTableWidgetItem(confidence);

                        // 将所有单元格的文本居中对齐
                        fileItem->setTextAlignment(Qt::AlignCenter);
                        statusItem->setTextAlignment(Qt::AlignCenter);
                        predItem->setTextAlignment(Qt::AlignCenter);
                        confItem->setTextAlignment(Qt::AlignCenter);

                        // 根据诊断结果设置醒目的字体颜色
                        if(status != "success") {
                            statusItem->setForeground(QBrush(Qt::red));
                            predItem->setForeground(QBrush(Qt::red));
                        } else if (prediction.contains("异常")) {
                            predItem->setForeground(QBrush(Qt::red));
                        } else {
                            predItem->setForeground(QBrush(Qt::darkGreen));
                        }

                        // 将创建好的单元格放入表格的对应位置
                        ui->tableWidget_results->setItem(i, 0, fileItem);
                        ui->tableWidget_results->setItem(i, 1, statusItem);
                        ui->tableWidget_results->setItem(i, 2, predItem);
                        ui->tableWidget_results->setItem(i, 3, confItem);
                    }

                    ui->label_status->setText(tr("诊断完成！"));
                    ui->progressBar->setValue(ui->progressBar->maximum());
                    QMessageBox::information(this, tr("完成"), tr("批量音频诊断分析已完成！"));
                } else if (rootObj.contains("continuous_result")) {
                    QJsonObject item = rootObj["continuous_result"].toObject();

                    // 获取当前表格的行数，并在最后面新插一行
                    int row = ui->tableWidget_results->rowCount();
                    ui->tableWidget_results->insertRow(row);

                    QString filename = item["filename"].toString();
                    QString status = item["status"].toString();
                    QString prediction = item.contains("prediction") ? item["prediction"].toString() : item["message"].toString();
                    QString confidence = item.contains("confidence") ? QString::number(item["confidence"].toDouble(), 'f', 4) : "-";

                    QTableWidgetItem *fileItem = new QTableWidgetItem(filename);
                    QTableWidgetItem *statusItem = new QTableWidgetItem(status == "success" ? "成功" : "失败");
                    QTableWidgetItem *predItem = new QTableWidgetItem(prediction);
                    QTableWidgetItem *confItem = new QTableWidgetItem(confidence);

                    fileItem->setTextAlignment(Qt::AlignCenter);
                    statusItem->setTextAlignment(Qt::AlignCenter);
                    predItem->setTextAlignment(Qt::AlignCenter);
                    confItem->setTextAlignment(Qt::AlignCenter);

                    // 设置颜色
                    if(status != "success" || prediction.contains("异常")) {
                        predItem->setForeground(QBrush(Qt::red));
                        statusItem->setForeground(QBrush(Qt::red));
                    } else {
                        predItem->setForeground(QBrush(Qt::darkGreen));
                    }

                    ui->tableWidget_results->setItem(row, 0, fileItem);
                    ui->tableWidget_results->setItem(row, 1, statusItem);
                    ui->tableWidget_results->setItem(row, 2, predItem);
                    ui->tableWidget_results->setItem(row, 3, confItem);

                    // 自动将表格滚动到最底部，实现“瀑布流”监控感
                    ui->tableWidget_results->scrollToBottom();
                }
            }
        }
        // 3. 其他 Python print 出来的调试信息或警告
        else {
            qDebug() << "Python 输出:" << line;
        }
    }

    // 如果进程已经结束（这里可以做个保护，防止按钮没恢复）
    if (diagnosisProcess->state() == QProcess::NotRunning) {
        ui->btn_start_diagnosis->setEnabled(true);
        ui->btn_start_diagnosis->setText(tr("开始诊断"));
    }
}


void MainWindow::on_btn_continuous_monitor_clicked()
{
    // 如果当前进程正在运行，说明用户想停止监控
    if (diagnosisProcess->state() == QProcess::Running) {
        diagnosisProcess->kill();          // 强制杀掉 Python 进程
        diagnosisProcess->waitForFinished();
        ui->btn_continuous_monitor->setText(tr("持续监控")); // 恢复按钮文字
        ui->btn_start_diagnosis->setEnabled(true);
        ui->label_status->setText(tr("监控已停止"));
        return;
    }

    // 如果没运行，说明用户想开始监控
    ui->tableWidget_results->setRowCount(0); // 清空旧表格
    ui->label_status->setText(tr("初始化模型环境中..."));

    QString modelPath = ui->lineEdit_model->text();
    QString folderPath = ui->lineEdit_folder->text();

    if (modelPath.isEmpty() || folderPath.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), tr("请先选择模型文件和待测试文件夹！"));
        return;
    }

    // 读取配置和设置路径
    QString configPath = QDir(QCoreApplication::applicationDirPath()).filePath("config.ini");
    QSettings settings(configPath, QSettings::IniFormat);
    QString pythonExe = settings.value("Environment/PythonExe", "").toString();
    QString algorithmDir = settings.value("Environment/AlgorithmDir", "").toString();

    QDir algoDir(algorithmDir);
    QString scriptPath = algoDir.absoluteFilePath("run_diagnosis.py");
    diagnosisProcess->setWorkingDirectory(algoDir.absolutePath());

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONIOENCODING", "utf-8");
    diagnosisProcess->setProcessEnvironment(env);

    // 组装参数，加上 --continuous
    QStringList arguments;
    arguments << "-u" << scriptPath << "--model_path" << modelPath << "--folder_path" << folderPath << "--continuous";

    // 禁用批量诊断按钮，把当前按钮变成停止
    ui->btn_start_diagnosis->setEnabled(false);
    ui->btn_continuous_monitor->setText(tr("停止监控"));

    diagnosisProcess->start(pythonExe, arguments);
}

