use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    format::Format,
    image::{ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, compute::ComputePipelineCreateInfo},
    sync::{self, GpuFuture},
};

use image::{ImageBuffer, Rgba};
use vulkano::image::Image;
use vulkano::image::view::ImageView;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};

use std::sync::Arc;

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("create instance failed");

    let physical = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices");

    for dev in physical {
        println!("Device {:?} found", dev.properties().device_name);
    }

    let phy_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("device not found");

    println!("Device {:?} chosen", phy_device.properties().device_name);

    for family in phy_device.queue_family_properties() {
        println!(
            "Find a queue family with {:?} queues with characteristic {:?}",
            family.queue_count, family.queue_flags
        );
    }

    let queue_family_index = phy_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("couldn't find a graphical queue family") as u32;

    println!("Find queue family {:?} with graphics", queue_family_index);

    let (device, mut queues) = Device::new(
        phy_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("fail to create device");

    // println!("Device created on {:?}, with {:?} queues",
    //     device.physical_device().properties().device_name, queues.count());

    let queue = queues.next().unwrap();

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "compute_pipeline/shaders/cshader.comp"
        }
    }

    let shader = cs::load(device.clone()).expect("failed to create shader module");

    let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").unwrap());
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .expect("layout creation error");

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("fail to create pipeline");

    let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = Image::new(
        mem_alloc.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            extent: [1024, 1024, 1],
            format: Format::R8G8B8A8_UNORM,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            ..Default::default()
        },
    )
    .expect("fail to create image");

    let view = ImageView::new_default(image.clone()).unwrap();

    let buf = Buffer::from_iter(
        mem_alloc.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("fail to create buffer");

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_alloc =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let set = PersistentDescriptorSet::new(
        &descriptor_set_alloc,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
        [],
    )
    .expect("fail to create persistent descriptor set");

    let command_alloc = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            ..Default::default()
        },
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_alloc,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("fail to create command buffer");

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .expect("fail to bind compute pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .expect("fail to bind descriptor sets")
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .expect("fail to set up the command buffer");

    let command_buffer = builder.build().expect("fail to build the command buffer");

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

    println!("Everything succeeded!");
}
